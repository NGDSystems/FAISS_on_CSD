/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_set_num_threads(int) {
    return 0;
}
inline int omp_get_num_threads() {
    return 1;
}
inline int omp_get_max_threads() {
    return 1;
}
inline int omp_get_thread_num() {
    return 0;
}
inline int omp_init_lock(int*) {
    return 0;
}
inline int omp_destroy_lock(int*) {
    return 0;
}
inline int omp_set_lock(int*) {
    return 0;
}
inline int omp_unset_lock(int*) {
    return 0;
}
inline int omp_set_nested(int) {
    return 0;
}
inline int omp_get_nested() {
    return 0;
}
inline int omp_in_parallel() {
    return 0;
}

#define omp_lock_t int
#endif
