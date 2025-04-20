#include <assert.h>
#include <pthread.h>
#include <stdbool.h>
#include <string.h>

#include "cilk-internal.h"
#include "fiber.h"
#include "frame.h"
#include "local-hypertable.h"
#include "scheduler.h"

hyper_table *scan_state_table;
pthread_mutex_t scan_state_table_lock;
bool scan_state_table_initialized = false;

typedef struct {
    size_t n;
    // TODO: fine-grained locking to actually increase parallelism.
    pthread_mutex_t lock;
    // Three cases of p = prefix_lens[i] (i is zero-based):
    // 1. p = 0: we don't have the local value for i yet.
    // 2. p < lowbit(i + 1): up[i] stores the sum of last p elements up to i.
    //    p should be a power of 2.
    // 3. p = lowbit(i + 1): up[i] and down[i] both store the sum of last p
    //    elements up to i.
    // 4. p = i + 1: down[i] stores the prefix sum of all elements up to i.
    //    up[i] stores the sum of last lowbit(i + 1) elements up to i.
    // TODO: make these arrays dynamically resizable.
    uint32_t *prefix_lens;
    void *up;
    void *down;
    __cilkrts_stack_frame **sfs;
} scan_state;

static inline void scan_init(scan_state *s, size_t size, size_t n) {
    s->n = n;
    s->prefix_lens = calloc(n, sizeof(uint32_t));
    s->up = malloc(n * size);
    s->down = malloc(n * size);
    s->sfs = calloc(n, sizeof(__cilkrts_stack_frame *));
    pthread_mutex_init(&s->lock, NULL);
}

static inline void scan_destroy(scan_state *s) {
    free(s->prefix_lens);
    free(s->up);
    free(s->down);
    free(s->sfs);
    pthread_mutex_destroy(&s->lock);
}

static inline size_t lowbit(size_t x) { return x & -x; }

// OpenCilk reducer use merge-left semantics, but for scan merge-right is more
// convenient.
// Note: this does not work for all types (e.g., if the type is
// self-referential).
__attribute__((always_inline)) static inline void
merge_right(void *left, void *right, void *temp, size_t size,
            __cilk_reduce_fn merge) {
    memcpy(temp, right, size);
    memcpy(right, left, size);
    merge(right, temp);
}

// If the task at idx has been suspended, wake it up as we now have the full
// prefix sum up to idx.
static void scan_wake_up(scan_state *s, size_t idx) {
    if (s->sfs[idx] == NULL)
        return; // This can happen if the calling task is commiting idx itself.
    __cilkrts_stack_frame *sf = s->sfs[idx];
    s->sfs[idx] = NULL;
    __cilkrts_worker *w = __cilkrts_get_tls_worker();

    // Is this the right way to resume?
    struct __cilkrts_stack_frame **tail =
        atomic_load_explicit(&w->tail, memory_order_relaxed);
    CILK_ASSERT((tail + 1) < w->ltq_limit);
    // store parent at *tail, and then increment tail
    *tail++ = sf;
    /* Release ordering ensures the two preceding stores are visible. */
    atomic_store_explicit(&w->tail, tail, memory_order_release);
}

// Try to have preceding nodes down sweep to the current node at idx if they are
// all available.
// Returns true if after the call, prefix_lens[idx] == idx + 1, which means we
// have the full prefix sum and that's stored in down[idx].
static bool scan_sweep_down_forward(scan_state *s, void *temp, size_t idx,
                                    size_t size, __cilk_reduce_fn merge) {
    if (s->prefix_lens[idx] == idx + 1)
        return true;
    size_t k = lowbit(idx + 1);
    if (!(s->prefix_lens[idx] & k))
        return false;
    // After upsweep, we already have in up[idx] the sum of last lowbit(idx + 1)
    // elements up to idx, so the node to merge to is idx - lowbit(idx + 1).
    size_t prev_idx = idx - k;
    if (scan_sweep_down_forward(s, temp, prev_idx, size, merge)) {
        merge_right((uint8_t *)s->down + prev_idx * size,
                    (uint8_t *)s->down + idx * size, temp, size, merge);
        s->prefix_lens[idx] = idx + 1;
        scan_wake_up(s, idx);
        return true;
    }
    return false;
}

// Given that we have prefix sum up to idx, sweep the result down to later
// nodes.
static void scan_sweep_down_backward(scan_state *s, void *temp, size_t idx,
                                     size_t size, __cilk_reduce_fn merge) {
    size_t k = lowbit(idx + 1);
    while (k >>= 1) {
        // TODO: these loop iterations can be parallelized.
        size_t next_idx = idx + k;
        if (next_idx < s->n && s->prefix_lens[next_idx] == k) {
            merge_right((uint8_t *)s->down + idx * size,
                        (uint8_t *)s->down + next_idx * size, temp, size,
                        merge);
            s->prefix_lens[next_idx] = next_idx + 1;
            scan_wake_up(s, next_idx);
            scan_sweep_down_backward(s, temp, next_idx, size, merge);
        }
    }
}

static void scan_sweep_down(scan_state *s, void *temp, size_t idx, size_t size,
                            __cilk_reduce_fn merge) {
    if (scan_sweep_down_forward(s, temp, idx, size, merge)) {
        scan_sweep_down_backward(s, temp, idx, size, merge);
    }
}

static void scan_sweep_up(scan_state *s, void *temp, size_t idx, size_t size,
                          __cilk_reduce_fn merge) {
    size_t k = 1;
    if ((idx + 1) & k) { // Equivalent to k == lowbit(idx + 1) here
        memcpy((uint8_t *)s->down + idx * size, (uint8_t *)s->up + idx * size,
               size);
        scan_sweep_down(s, temp, idx, size, merge);
    }
    // Loop invariant: prefix_lens[idx] & k != 0.
    // nidx = idx ^ k will be the neighbor of idx in the tree at this level.
    // Consider the possible values of prefix_lens[nidx]:
    // * 0: we don't have the local value for neighbor yet. Can't merge.
    // * < k: neighbor hasn't been pushed up to our level yet. Can't merge.
    // * k: neighbor has been pushed up to our level. Merge.
    // * > k but <= lowbit(nidx + 1): can't ever happen.
    // * ndix + 1: this can only happen if we are the right neighbor. The left
    //   neighbor already has all the prefix sum. up[ndix] stores the sum of the
    //   last lowbit(nidx + 1) elements up to nidx. Merge.
    // The condition is basically (p & k != 0).
    while ((idx ^ k) < s->n && (s->prefix_lens[idx ^ k] & k)) {
        size_t idx_left = idx & ~k;
        size_t idx_right = idx | k;
        merge_right((uint8_t *)s->up + idx_left * size,
                    (uint8_t *)s->up + idx_right * size, temp, size, merge);
        s->prefix_lens[idx = idx_right] = k <<= 1;
        if (s->prefix_lens[idx] == idx + 1)
            scan_wake_up(s, idx);
        if ((idx + 1) & k) { // Equivalent to k == lowbit(idx + 1) here
            memcpy((uint8_t *)s->down + idx * size,
                   (uint8_t *)s->up + idx * size, size);
            scan_sweep_down(s, temp, idx, size, merge);
        }
    }
}

static void scan_commit(scan_state *s, void *view, size_t idx, size_t size,
                        __cilk_reduce_fn merge) {
    CILK_ASSERT(idx < s->n && "Index out of bounds");
    pthread_mutex_lock(&s->lock);
    CILK_ASSERT(s->prefix_lens[idx] == 0 &&
                "Scan state already committed at this index");
    memcpy((uint8_t *)s->up + idx * size, view, size);
    s->prefix_lens[idx] = 1;
    scan_sweep_up(s, view, idx, size, merge);
    if (s->prefix_lens[idx] == idx + 1) {
        // Fast path: no need to suspend as we already have the full prefix sum.
        memcpy(view, (uint8_t *)s->down + idx * size, size);
        pthread_mutex_unlock(&s->lock);
        return;
    }

    // Don't have the full prefix sum yet. Suspend the current task.
    __cilkrts_stack_frame *sf = __cilkrts_current_fh->current_stack_frame;
    sysdep_save_fp_ctrl_state(sf);
    if (__builtin_setjmp(sf->ctx)) {
        CILK_ASSERT(s->prefix_lens[idx] == idx + 1);
        // This is done without holding the lock, but it's fine, once
        // s->prefix_lens[idx] == idx + 1, s->down[idx] will not be modified
        // anymore.
        memcpy(view, (uint8_t *)s->down + idx * size, size);
    } else {
        CILK_ASSERT(
            s->sfs[idx] == NULL &&
            "Scan state already suspended"); // How can this be possible?
        s->sfs[idx] = sf;
        pthread_mutex_unlock(&s->lock);
        // Is this the right way to suspend?
        longjmp_to_runtime(get_worker_from_stack(sf));
    }
}

void __cilkrts_init_scan_state_table() {
    CILK_ASSERT(!scan_state_table_initialized &&
                "Scan state table already initialized");
    pthread_mutex_init(&scan_state_table_lock, NULL);
    scan_state_table = __cilkrts_local_hyper_table_alloc();
    scan_state_table_initialized = true;
}

void __cilkrts_destroy_scan_state_table() {
    CILK_ASSERT(scan_state_table_initialized &&
                "Scan state table not initialized");
    scan_state_table_initialized = false;
    local_hyper_table_free(scan_state_table);
    scan_state_table = NULL;
    pthread_mutex_destroy(&scan_state_table_lock);
}

void __cilkrts_scanner_commit(void *key, void *view, size_t idx, size_t size,
                               void *reduce) {
    CILK_ASSERT(scan_state_table_initialized &&
                "Scan state table not initialized");
    pthread_mutex_lock(&scan_state_table_lock);
    struct bucket *b = find_hyperobject(scan_state_table, (uintptr_t)key);
    CILK_ASSERT(b != NULL && "Key not registered");
    pthread_mutex_unlock(&scan_state_table_lock);
    scan_state *s = (scan_state *)b->value.view;
    scan_commit(s, view, idx, size, (__cilk_reduce_fn)reduce);
}

void __cilkrts_scanner_register(void *key, size_t n, size_t size,
                                __cilk_reduce_fn reduce) {
    CILK_ASSERT(scan_state_table_initialized &&
                "Scan state table not initialized");
    pthread_mutex_lock(&scan_state_table_lock);
    struct bucket *b = find_hyperobject(scan_state_table, (uintptr_t)key);
    CILK_ASSERT(b == NULL && "Key already registered");
    void *new_scan_state = malloc(sizeof(scan_state));
    struct bucket new_bucket = {
        .key = (uintptr_t)key,
        .value = {.view = new_scan_state, .reduce_fn = reduce}};
    bool success = insert_hyperobject(scan_state_table, new_bucket);
    CILK_ASSERT(success && "Failed to insert new scan state");
    (void)success;
    pthread_mutex_unlock(&scan_state_table_lock);
    scan_init((scan_state *)new_scan_state, size, n);
}

void __cilkrts_scanner_unregister(void *key) {
    CILK_ASSERT(scan_state_table_initialized &&
                "Scan state table not initialized");
    pthread_mutex_lock(&scan_state_table_lock);
    struct bucket *b = find_hyperobject(scan_state_table, (uintptr_t)key);
    CILK_ASSERT(b != NULL && "Key not registered");
    scan_state *s = (scan_state *)b->value.view;
    bool success = remove_hyperobject(scan_state_table, (uintptr_t)key);
    CILK_ASSERT(success && "Failed to unregister scan state");
    (void)success;
    pthread_mutex_unlock(&scan_state_table_lock);
    scan_destroy(s);
}
