#include <chrono>
#include <iostream>

#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"
#include "noarr/structures/interop/bag.hpp"
#include "noarr/structures/interop/serialize_data.hpp"

using num_t = float;

namespace {

// initialization function
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) {
    // C: i x j
    // A: i x k
    // B: k x j

    alpha = 1.5;
    beta = 1.2;

    noarr::traverser(C)
        .for_each([=](auto state) {
            auto [i, j] = noarr::get_indices<'i', 'j'>(state);
            C[state] = (num_t)((i * j + 1) % (C | noarr::get_length<'i'>())) / (C | noarr::get_length<'i'>());
        });

    noarr::traverser(A)
        .for_each([=](auto state) {
            auto [i, k] = noarr::get_indices<'i', 'k'>(state);
            A[state] = (num_t)(i * (k + 1) % (A | noarr::get_length<'k'>())) / (A | noarr::get_length<'k'>());
        });
    
    noarr::traverser(B)
        .for_each([=](auto state) {
            auto [k, j] = noarr::get_indices<'k', 'j'>(state);
            B[state] = (num_t)(k * (j + 2) % (B | noarr::get_length<'j'>())) / (B | noarr::get_length<'j'>());
        });
}

// computation kernel
void kernel_gemm(num_t alpha, num_t beta, auto C, auto A, auto B) {
    // C: i x j
    // A: i x k
    // B: k x j

    noarr::traverser(C)
        .for_each([=](auto state) {
            C[state] *= beta;
        });

    noarr::traverser(C, A, B)
        .for_each([=](auto state) {
            C[state] += alpha * A[state] * B[state];
        });
}

} // namespace

int main(int argc, char *argv[]) {
    using namespace std::string_literals;

    // problem size
    std::size_t ni = NI;
    std::size_t nj = NJ;
    std::size_t nk = NK;

    // input data
    num_t alpha;
    num_t beta;

    auto C = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(ni, nj));
    auto A = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'k'>(ni, nk));
    auto B = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'k', 'j'>(nk, nj));

    // initialize data
    init_array(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

    auto start = std::chrono::high_resolution_clock::now();

    // run kernel
    kernel_gemm(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // print results
    if (argv[0] != ""s)
        noarr::serialize_data(std::cout, C.get_ref());

    std::cerr << duration << std::endl;
}
