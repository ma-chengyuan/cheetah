#ifndef CLANGD
#include <cilk/cilk.h>
#else
#define cilk_spawn
#define cilk_sync
#define cilk_for for
#define cilk_reducer(id, merge)
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void __cilkrts_scanner_commit(void *key, void *view, size_t idx, size_t size,
                              void *reduce);
void __cilkrts_scanner_register(void *key, size_t n, size_t size, void *reduce);
void __cilkrts_scanner_unregister(void *key);

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
    if (rr2->start == -1)
        return;
    if (rr1->start == -1) {
        *rr1 = *rr2;
        return;
    }
    if (rr1->end != rr2->start) {
        fprintf(stderr, "Invalid merge of [%d, %d), [%d, %d)\n", rr1->start,
                rr1->end, rr2->start, rr2->end);
        abort();
    }
    rr1->end = rr2->end;
}

__attribute__((noinline)) void test(void) {
    range /*cilk_scanner(id_range, merge_range)*/ scanner_base;
    size_t n = 8;

    // This is similar to __cilkrts_reducer_register, but takes an extra
    // argument n denoting the "extent" of the scan. With some careful
    // programming, n may not be necessary and can be dynamically adjusted if
    // needed.
    __cilkrts_scanner_register(&scanner_base, n, sizeof(range), merge_range);
#pragma cilk grainsize 1
    cilk_for (size_t i = 0; i < n; i++) {
        // This should be inserted by the compiler, similar to
        // __cilkrts_reducer_lookup(). Unlike reducer, each loop iteration is
        // guaranteed an independent view at the beginning.
        range local_view;
        id_range(&local_view);

        size_t idx = i;
        // Now modify the local view.
        local_view.start = (uint32_t)idx;
        local_view.end = (uint32_t)idx + 1;

        // "Commit" our changes to the local view to the scan. The third
        // argument, idx should conceptually be the number of commits before
        // this one in the serial projection of the program since the last
        // __cilkrts_scanner_register. I don't know how to get Cheetah to track
        // this automatically though.
        __cilkrts_scanner_commit(&scanner_base, &local_view, idx, sizeof(range),
                                 merge_range);
        // Implicitly, the call above suspends the current task until all tasks
        // with smaller idx reach the commit point and the inclusive prefix sum
        // at idx is available and copied to local_view. Importantly, I think
        // the commit call should never wait for commits with higher idx. 

        // This commit call maps very nicely to the decoupled lookback algorithm
        // on GPU, but does not really fit into the fork-join paradigm...

        // I tried using idx = n - i - 1 to simulate when commits happen out of order.
        // It does not work for now, because I don't know cheetah well enough...
        printf("scanner[%zu] = [%d, %d)\n", idx, local_view.start, local_view.end);
    }
    // This is similar to __cilkrts_reducer_unregister
    __cilkrts_scanner_unregister(&scanner_base);
}

int main(int argc, char *argv[]) {
    // Temporary hack to ensure test runs in a cilkified region.
    cilk_spawn { test(); }
    cilk_sync;
    return 0;
}
