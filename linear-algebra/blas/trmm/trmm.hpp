#ifndef TRMM_HPP
#define TRMM_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define NI 20
# define NJ 30
#elif defined(SMALL_DATASET)
# define NI 60
# define NJ 80
#elif defined(MEDIUM_DATASET)
# define NI 200
# define NJ 240
#elif defined(LARGE_DATASET)
# define NI 1000
# define NJ 1200
#elif defined(EXTRALARGE_DATASET)
# define NI 2000
# define NJ 2600
#endif

#endif // TRMM_HPP
