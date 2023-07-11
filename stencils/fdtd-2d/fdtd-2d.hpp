#ifndef FDTD_2D_HPP
#define FDTD_2D_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define TMAX 20
# define NI 20
# define NJ 30
#elif defined(SMALL_DATASET)
# define TMAX 40
# define NI 60
# define NJ 80
#elif defined(MEDIUM_DATASET)
# define TMAX 100
# define NI 200
# define NJ 240
#elif defined(LARGE_DATASET)
# define TMAX 500
# define NI 1000
# define NJ 1200
#elif defined(EXTRALARGE_DATASET)
# define TMAX 1000
# define NI 2000
# define NJ 2600
#endif

#endif // FDTD_2D_HPP
