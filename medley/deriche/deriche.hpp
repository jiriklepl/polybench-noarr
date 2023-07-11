#ifndef DERICHE_HPP
#define DERICHE_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define NW 64
# define NH 64
#elif defined(SMALL_DATASET)
# define NW 192
# define NH 128
#elif defined(MEDIUM_DATASET)
# define NW 720
# define NH 480
#elif defined(LARGE_DATASET)
# define NW 4096
# define NH 2160
#elif defined(EXTRALARGE_DATASET)
# define NW 7680
# define NH 4320
#endif

#endif // DERICHE_HPP
