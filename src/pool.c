// A fixed team of threads. pool_run(fn, arg) calls fn(arg, i, n) once on each
// of the n threads (the caller is thread 0) and returns when all are done.
// A forward pass hands the team a few hundred small jobs per token, so idle
// workers spin on a counter instead of sleeping: waking a sleeping thread
// costs more than most of the jobs.
#include "abacus.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif

static Task task;
static void *task_arg;
static int n_threads = 1;
static atomic_int generation, pending;

static inline void cpu_relax(void) {
#if defined(__aarch64__)
    __asm__ volatile("yield");
#elif defined(__x86_64__)
    __asm__ volatile("pause");
#endif
}

static void *worker(void *p) {
    int id = (int)(intptr_t)p, seen = 0;
    for (;;) {
        while (atomic_load_explicit(&generation, memory_order_acquire) == seen) cpu_relax();
        seen++;
        task(task_arg, id, n_threads);
        atomic_fetch_sub_explicit(&pending, 1, memory_order_release);
    }
    return NULL;
}

int default_threads(void) {
#ifdef __linux__
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof set, &set) == 0) return CPU_COUNT(&set);
#endif
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
}

void pool_start(int n) {
    n_threads = n > 1 ? n : 1;
    for (int i = 1; i < n; i++) {
        pthread_t t;
        pthread_create(&t, NULL, worker, (void *)(intptr_t)i);
        pthread_detach(t);
    }
}

int pool_size(void) { return n_threads; }

void pool_run(Task fn, void *arg) {
    task = fn, task_arg = arg;
    atomic_store_explicit(&pending, n_threads - 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&generation, 1, memory_order_release);
    fn(arg, 0, n_threads);
    while (atomic_load_explicit(&pending, memory_order_acquire)) cpu_relax();
}
