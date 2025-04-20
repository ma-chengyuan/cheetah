/**
 * A push-based implementation of Belloch's work-efficient scan algorithm.
 */

#include <assert.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*__cilk_reduce_fn)(void *, void *);
typedef void (*__cilk_identity_fn)(void *);

// Some bithacks:
static inline size_t lowbit(size_t x) { return x & -x; }

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
} scan_state;

void scan_init(scan_state *s, size_t size, size_t n) {
    s->n = n;
    s->prefix_lens = (uint32_t *)calloc(n, sizeof(uint32_t));
    s->up = malloc(n * size);
    s->down = malloc(n * size);
    pthread_mutex_init(&s->lock, NULL);
}

void scan_destroy(scan_state *s) {
    free(s->prefix_lens);
    free(s->up);
    free(s->down);
    pthread_mutex_destroy(&s->lock);
}

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

// Try to have preceding nodes down sweep to the current node at idx if they are
// all available.
// Returns true if after the call, prefix_lens[idx] == idx + 1, which means we
// have the full prefix sum and that's stored in down[idx].
bool scan_sweep_down_forward(scan_state *s, void *temp, size_t idx, size_t size,
                             __cilk_reduce_fn merge) {
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
        return true;
    }
    return false;
}

// Given that we have prefix sum up to idx, sweep the result down to later
// nodes.
void scan_sweep_down_backward(scan_state *s, void *temp, size_t idx,
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
            scan_sweep_down_backward(s, temp, next_idx, size, merge);
        }
    }
}

void scan_sweep_down(scan_state *s, void *temp, size_t idx, size_t size,
                     __cilk_reduce_fn merge) {
    if (scan_sweep_down_forward(s, temp, idx, size, merge)) {
        scan_sweep_down_backward(s, temp, idx, size, merge);
    }
}

void scan_sweep_up(scan_state *s, void *temp, size_t idx, size_t size,
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
        if ((idx + 1) & k) { // Equivalent to k == lowbit(idx + 1) here
            memcpy((uint8_t *)s->down + idx * size,
                   (uint8_t *)s->up + idx * size, size);
            scan_sweep_down(s, temp, idx, size, merge);
        }
    }
}

void scan_commit(scan_state *s, void *view, size_t idx, size_t size,
                 __cilk_reduce_fn merge) {
    memcpy((uint8_t *)s->up + idx * size, view, size);
    s->prefix_lens[idx] = 1;
    pthread_mutex_lock(&s->lock);
    scan_sweep_up(s, view, idx, size, merge);
    pthread_mutex_unlock(&s->lock);
}

// A test program to check the correctness of the scan implementation.

typedef struct {
    int32_t start;
    int32_t end;
} range;

void id_range(void *r) {
    range *rr = (range *)r;
    rr->start = rr->end = -1;
}

void merge_range(void *r1, void *r2) {
    range *rr1 = (range *)r1;
    range *rr2 = (range *)r2;
    if (rr2->start == -1) {
        return;
    }
    if (rr1->start == -1) {
        *rr1 = *rr2;
        return;
    }
    // assert(rr1->end == rr2->start && "Invalid merge");
    if (rr1->end != rr2->start) {
        fprintf(stderr, "Invalid merge of [%d, %d), [%d, %d)\n", rr1->start,
                rr1->end, rr2->start, rr2->end);
        abort();
    }
    rr1->end = rr2->end;
}

bool next_permutation(size_t *begin, size_t *end);

int main() {
    scan_state s;
    size_t n = 10;
    // Let's try all different commit order and ensures that the push-based
    // algorithm works for all of them.
    size_t *order = (size_t *)malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) {
        order[i] = i;
    }
    bool *commiteed = (bool *)calloc(n, sizeof(bool));
    do {
        scan_init(&s, sizeof(range), n);
        memset(commiteed, 0, n * sizeof(bool));
        for (size_t i = 0; i < n; i++) {
            size_t idx = order[i];
            range r;
            r.start = idx;
            r.end = idx + 1;
            scan_commit(&s, &r, idx, sizeof(range), merge_range);

            commiteed[idx] = true;
            bool all_commited_before = true;
            for (size_t j = 0; j < idx; j++)
                all_commited_before &= commiteed[j];
            if (all_commited_before) {
                range *r = (range *)((uint8_t *)s.down + idx * sizeof(range));
                if (r->start == 0 && r->end == idx + 1) {
                    // If scan_commit has been called for all previous indices,
                    // then we should have the prefix sum at idx at this point.
                    continue;
                }

                fprintf(stderr, "buggy scan order:");
                for (size_t j = 0; j < n; j++)
                    fprintf(stderr, " %zu", order[j]);
                fprintf(stderr, "\n");
                for (size_t j = 0; j < n; j++) {
                    range *r = (range *)((uint8_t *)s.up + j * sizeof(range));
                    fprintf(stderr, "up[%zu] = [%d, %d)\n", j, r->start,
                            r->end);
                }
                for (size_t j = 0; j < n; j++) {
                    range *r = (range *)((uint8_t *)s.down + j * sizeof(range));
                    fprintf(stderr, "down[%zu] = [%d, %d)\n", j, r->start,
                            r->end);
                }
                scan_destroy(&s);
                free(commiteed);
                free(order);
                return 1;
            }
        }
        scan_destroy(&s);
    } while (next_permutation(order, order + n));

    free(commiteed);
    free(order);
    return 0;
}

void reverse(size_t *begin, size_t *end) {
    size_t *i = begin;
    --end;
    while (i < end) {
        size_t temp = *i;
        *i++ = *end;
        *end-- = temp;
    }
}

bool next_permutation(size_t *begin, size_t *end) {
    size_t *i = end;
    if (begin == end || begin + 1 == end)
        return false;
    --i;
    while (true) {
        size_t *j = i;
        --i;
        if (*i < *j) {
            size_t *k = end;
            while (*i >= *(--k))
                ;
            size_t temp = *i;
            *i = *k;
            *k = temp;
            reverse(j, end);
            return true;
        }
        if (i == begin) {
            reverse(begin, end);
            return false;
        }
    }
}