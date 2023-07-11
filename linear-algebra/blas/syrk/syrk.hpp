#ifndef SYRK_HPP
#define SYRK_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define NI 30
# define NK 20
#elif defined(SMALL_DATASET)
# define NI 80
# define NK 60
#elif defined(MEDIUM_DATASET)
# define NI 240
# define NK 200
#elif defined(LARGE_DATASET)
# define NI 1200
# define NK 1000
#elif defined(EXTRALARGE_DATASET)
# define NI 2600
# define NK 2000
#endif

#endif // SYRK_HPP
