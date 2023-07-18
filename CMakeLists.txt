cmake_minimum_required(VERSION 3.10)
Include(FetchContent)

# set the project name
project(NoarrPolybench VERSION 0.0.1)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# download testing library Catch2
FetchContent_Declare(
  Noarr
  GIT_REPOSITORY https://github.com/jiriklepl/noarr-structures.git
  GIT_TAG        constraints-WIP)
FetchContent_MakeAvailable(Noarr)

# datamining
add_executable(correlation datamining/correlation/correlation.cpp)
add_executable(covariance datamining/covariance/covariance.cpp)

target_include_directories(correlation PRIVATE include)
target_include_directories(correlation PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(covariance PRIVATE include)
target_include_directories(covariance PRIVATE ${Noarr_SOURCE_DIR}/include)


# linear algebra - blas
add_executable(gemm linear-algebra/blas/gemm/gemm.cpp)
add_executable(gemver linear-algebra/blas/gemver/gemver.cpp)
add_executable(gesummv linear-algebra/blas/gesummv/gesummv.cpp)
add_executable(symm linear-algebra/blas/symm/symm.cpp)
add_executable(syr2k linear-algebra/blas/syr2k/syr2k.cpp)
add_executable(syrk linear-algebra/blas/syrk/syrk.cpp)
add_executable(trmm linear-algebra/blas/trmm/trmm.cpp)

target_include_directories(gemm PRIVATE include)
target_include_directories(gemm PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(gemver PRIVATE include)
target_include_directories(gemver PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(gesummv PRIVATE include)
target_include_directories(gesummv PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(symm PRIVATE include)
target_include_directories(symm PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(syr2k PRIVATE include)
target_include_directories(syr2k PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(syrk PRIVATE include)
target_include_directories(syrk PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(trmm PRIVATE include)
target_include_directories(trmm PRIVATE ${Noarr_SOURCE_DIR}/include)


# linear algebra - kernels
add_executable(2mm linear-algebra/kernels/2mm/2mm.cpp)
add_executable(3mm linear-algebra/kernels/3mm/3mm.cpp)
add_executable(atax linear-algebra/kernels/atax/atax.cpp)
add_executable(bicg linear-algebra/kernels/bicg/bicg.cpp)
add_executable(doitgen linear-algebra/kernels/doitgen/doitgen.cpp)
add_executable(mvt linear-algebra/kernels/mvt/mvt.cpp)

target_include_directories(2mm PRIVATE include)
target_include_directories(2mm PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(3mm PRIVATE include)
target_include_directories(3mm PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(atax PRIVATE include)
target_include_directories(atax PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(bicg PRIVATE include)
target_include_directories(bicg PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(doitgen PRIVATE include)
target_include_directories(doitgen PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(mvt PRIVATE include)
target_include_directories(mvt PRIVATE ${Noarr_SOURCE_DIR}/include)


# linear algebra - solvers
add_executable(cholesky linear-algebra/solvers/cholesky/cholesky.cpp)
add_executable(durbin linear-algebra/solvers/durbin/durbin.cpp)
add_executable(gramschmidt linear-algebra/solvers/gramschmidt/gramschmidt.cpp)
add_executable(lu linear-algebra/solvers/lu/lu.cpp)
add_executable(ludcmp linear-algebra/solvers/ludcmp/ludcmp.cpp)
add_executable(trisolv linear-algebra/solvers/trisolv/trisolv.cpp)

target_include_directories(cholesky PRIVATE include)
target_include_directories(cholesky PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(durbin PRIVATE include)
target_include_directories(durbin PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(gramschmidt PRIVATE include)
target_include_directories(gramschmidt PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(lu PRIVATE include)
target_include_directories(lu PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(ludcmp PRIVATE include)
target_include_directories(ludcmp PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(trisolv PRIVATE include)
target_include_directories(trisolv PRIVATE ${Noarr_SOURCE_DIR}/include)


# medley
add_executable(deriche medley/deriche/deriche.cpp)
add_executable(floyd-warshall medley/floyd-warshall/floyd-warshall.cpp)
add_executable(nussinov medley/nussinov/nussinov.cpp)

target_include_directories(deriche PRIVATE include)
target_include_directories(deriche PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(floyd-warshall PRIVATE include)
target_include_directories(floyd-warshall PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(nussinov PRIVATE include)
target_include_directories(nussinov PRIVATE ${Noarr_SOURCE_DIR}/include)


# stencils
add_executable(adi stencils/adi/adi.cpp)
add_executable(fdtd-2d stencils/fdtd-2d/fdtd-2d.cpp)
add_executable(heat-3d stencils/heat-3d/heat-3d.cpp)
add_executable(jacobi-1d stencils/jacobi-1d/jacobi-1d.cpp)
add_executable(jacobi-2d stencils/jacobi-2d/jacobi-2d.cpp)
add_executable(seidel-2d stencils/seidel-2d/seidel-2d.cpp)

target_include_directories(adi PRIVATE include)
target_include_directories(adi PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(fdtd-2d PRIVATE include)
target_include_directories(fdtd-2d PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(heat-3d PRIVATE include)
target_include_directories(heat-3d PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(jacobi-1d PRIVATE include)
target_include_directories(jacobi-1d PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(jacobi-2d PRIVATE include)
target_include_directories(jacobi-2d PRIVATE ${Noarr_SOURCE_DIR}/include)
target_include_directories(seidel-2d PRIVATE include)
target_include_directories(seidel-2d PRIVATE ${Noarr_SOURCE_DIR}/include)
