#ifndef ATAX_HPP
#define ATAX_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define NI 38
# define NJ 42
#elif defined(SMALL_DATASET)
# define NI 116
# define NJ 124
#elif defined(MEDIUM_DATASET)
# define NI 390
# define NJ 410
#elif defined(LARGE_DATASET)
# define NI 1900
# define NJ 2100
#elif defined(EXTRALARGE_DATASET)
# define NI 1800
# define NJ 2200
#endif

#endif // ATAX_HPP
