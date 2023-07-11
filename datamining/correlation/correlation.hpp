#ifndef CORRELATION_HPP
#define CORRELATION_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define NK 32
# define NJ 28
#elif defined(SMALL_DATASET)
# define NK 100
# define NJ 80
#elif defined(MEDIUM_DATASET)
# define NK 260
# define NJ 240
#elif defined(LARGE_DATASET)
# define NK 1400
# define NJ 1200
#elif defined(EXTRALARGE_DATASET)
# define NK 3000
# define NJ 2600
#endif

#endif // CORRELATION_HPP
