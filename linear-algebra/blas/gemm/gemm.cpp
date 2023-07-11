#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

// initialization function
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) {
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

int main() { /* placeholder */}
