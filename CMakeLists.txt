cmake_minimum_required(VERSION 3.10)
Include(FetchContent)

# set the project name
project(NoarrPolybench VERSION 0.0.1)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# download Noarr
FetchContent_Declare(
  Noarr
  GIT_REPOSITORY https://github.com/jiriklepl/noarr-structures.git
  GIT_TAG        tuning)
FetchContent_MakeAvailable(Noarr)

include_directories(include)
include_directories(${Noarr_SOURCE_DIR}/include)
include_directories(${Noarr_SOURCE_DIR}/autotuners/opentuner_test)

# datamining
add_executable(correlation datamining/correlation/correlation.cpp)
add_executable(covariance datamining/covariance/covariance.cpp)

# linear algebra - blas
add_executable(gemm linear-algebra/blas/gemm/gemm.cpp)
add_executable(gemver linear-algebra/blas/gemver/gemver.cpp)
add_executable(gesummv linear-algebra/blas/gesummv/gesummv.cpp)
add_executable(symm linear-algebra/blas/symm/symm.cpp)
add_executable(syr2k linear-algebra/blas/syr2k/syr2k.cpp)
add_executable(syrk linear-algebra/blas/syrk/syrk.cpp)
add_executable(trmm linear-algebra/blas/trmm/trmm.cpp)

# linear algebra - kernels
add_executable(2mm linear-algebra/kernels/2mm/2mm.cpp)
add_executable(3mm linear-algebra/kernels/3mm/3mm.cpp)
add_executable(atax linear-algebra/kernels/atax/atax.cpp)
add_executable(bicg linear-algebra/kernels/bicg/bicg.cpp)
add_executable(doitgen linear-algebra/kernels/doitgen/doitgen.cpp)
add_executable(mvt linear-algebra/kernels/mvt/mvt.cpp)

# linear algebra - solvers
add_executable(cholesky linear-algebra/solvers/cholesky/cholesky.cpp)
add_executable(durbin linear-algebra/solvers/durbin/durbin.cpp)
add_executable(gramschmidt linear-algebra/solvers/gramschmidt/gramschmidt.cpp)
add_executable(lu linear-algebra/solvers/lu/lu.cpp)
add_executable(ludcmp linear-algebra/solvers/ludcmp/ludcmp.cpp)
add_executable(trisolv linear-algebra/solvers/trisolv/trisolv.cpp)

# medley
add_executable(deriche medley/deriche/deriche.cpp)
add_executable(floyd-warshall medley/floyd-warshall/floyd-warshall.cpp)
add_executable(nussinov medley/nussinov/nussinov.cpp)

# stencils
add_executable(adi stencils/adi/adi.cpp)
add_executable(fdtd-2d stencils/fdtd-2d/fdtd-2d.cpp)
add_executable(heat-3d stencils/heat-3d/heat-3d.cpp)
add_executable(jacobi-1d stencils/jacobi-1d/jacobi-1d.cpp)
add_executable(jacobi-2d stencils/jacobi-2d/jacobi-2d.cpp)
add_executable(seidel-2d stencils/seidel-2d/seidel-2d.cpp)

# AUTOTUNING

add_executable(floyd-warshall_autotune medley/floyd-warshall/floyd-warshall.cpp)
target_compile_definitions(floyd-warshall_autotune PRIVATE NOARR_TUNE)

add_executable(gemm_autotune linear-algebra/blas/gemm/gemm.cpp)
target_compile_definitions(gemm_autotune PRIVATE NOARR_TUNE)

add_executable(gemver_autotune linear-algebra/blas/gemver/gemver.cpp)
target_compile_definitions(gemver_autotune PRIVATE NOARR_TUNE)

add_executable(mvt_autotune linear-algebra/kernels/mvt/mvt.cpp)
target_compile_definitions(mvt_autotune PRIVATE NOARR_TUNE)

# TODO: fix 2mm autotuning (it takes too long)
# add_executable(2mm_autotune linear-algebra/kernels/2mm/2mm.cpp)
# target_compile_definitions(2mm_autotune PRIVATE NOARR_TUNE)

# TODO: fix 3mm autotuning (it takes too long)
# add_executable(3mm_autotune linear-algebra/kernels/3mm/3mm.cpp)
# target_compile_definitions(3mm_autotune PRIVATE NOARR_TUNE)

add_executable(doitgen_autotune linear-algebra/kernels/doitgen/doitgen.cpp)
target_compile_definitions(doitgen_autotune PRIVATE NOARR_TUNE)

add_executable(heat-3d_autotune stencils/heat-3d/heat-3d.cpp)
target_compile_definitions(heat-3d_autotune PRIVATE NOARR_TUNE)

add_executable(jacobi-2d_autotune stencils/jacobi-2d/jacobi-2d.cpp)
target_compile_definitions(jacobi-2d_autotune PRIVATE NOARR_TUNE)

add_executable(atax_autotune linear-algebra/kernels/atax/atax.cpp)
target_compile_definitions(atax_autotune PRIVATE NOARR_TUNE)

# TODO: add autotuning for correlation
# add_executable(correlation_autotune datamining/correlation/correlation.cpp)
# target_compile_definitions(correlation_autotune PRIVATE NOARR_TUNE)

add_executable(covariance_autotune datamining/covariance/covariance.cpp)
target_compile_definitions(covariance_autotune PRIVATE NOARR_TUNE)

add_executable(gesummv_autotune linear-algebra/blas/gesummv/gesummv.cpp)
target_compile_definitions(gesummv_autotune PRIVATE NOARR_TUNE)

add_executable(syr2k_autotune linear-algebra/blas/syr2k/syr2k.cpp)
target_compile_definitions(syr2k_autotune PRIVATE NOARR_TUNE)

add_executable(syrk_autotune linear-algebra/blas/syrk/syrk.cpp)
target_compile_definitions(syrk_autotune PRIVATE NOARR_TUNE)
