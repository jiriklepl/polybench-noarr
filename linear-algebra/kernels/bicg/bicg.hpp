#ifndef BICG_HPP
#define BICG_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define NI 42
# define NJ 38
#elif defined(SMALL_DATASET)
# define NI 124
# define NJ 116
#elif defined(MEDIUM_DATASET)
# define NI 410
# define NJ 390
#elif defined(LARGE_DATASET)
# define NI 2100
# define NJ 1900
#elif defined(EXTRALARGE_DATASET)
# define NI 2200
# define NJ 1800
#endif

#endif // BICG_HPP
