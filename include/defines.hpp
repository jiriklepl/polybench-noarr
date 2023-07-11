#ifndef DEFINES_HPP
#define DEFINES_HPP

#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
# error "Please define one of MINI_DATASET, SMALL_DATASET, MEDIUM_DATASET, LARGE_DATASET, EXTRALARGE_DATASET"
# define MINI_DATASET
#endif

#if !defined(DATA_TYPE_IS_INT) || !defined(DATA_TYPE_IS_FLOAT) || !defined(DATA_TYPE_IS_DOUBLE)
# error "Please define one of DATA_TYPE_IS_INT, DATA_TYPE_IS_FLOAT, DATA_TYPE_IS_DOUBLE"
# define DATA_TYPE_IS_FLOAT
#endif

#ifdef DATA_TYPE_IS_INT
# define DATA_TYPE int
#elif defined(DATA_TYPE_IS_FLOAT)
# define DATA_TYPE float
#elif defined(DATA_TYPE_IS_DOUBLE)
# define DATA_TYPE double
#endif

#endif // DEFINES_HPP