#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_gemm(num_t alpha, num_t beta, auto C, auto A, auto B) {
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
