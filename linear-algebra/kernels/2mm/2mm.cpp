#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

// initialization function
void init_array(num_t &alpha, num_t &beta, auto A, auto B, auto C, auto D) {
    // tmp: i x j
    // A: i x k
    // B: k x j
    // C: j x l
    // D: i x l

    alpha = 1.5;
    beta = 1.2;

    auto ni = A | noarr::get_length<'i'>();
    auto nj = B | noarr::get_length<'j'>();
    auto nk = A | noarr::get_length<'k'>();
    auto nl = C | noarr::get_length<'l'>();

    noarr::traverser(A)
        .for_each([=](auto state) {
            auto [i, k] = noarr::get_indices<'i', 'k'>(state);
            A[state] = (num_t)((i * k + 1) % ni) / ni;
        });

    noarr::traverser(B)
        .for_each([=](auto state) {
            auto [k, j] = noarr::get_indices<'k', 'j'>(state);
            B[state] = (num_t)(k * (j + 1) % nj) / nj;
        });

    noarr::traverser(C)
        .for_each([=](auto state) {
            auto [j, l] = noarr::get_indices<'j', 'l'>(state);
            C[state] = (num_t)((j * (l + 3) + 1) % nl) / nl;
        });

    noarr::traverser(D)
        .for_each([=](auto state) {
            auto [i, l] = noarr::get_indices<'i', 'l'>(state);
            D[state] = (num_t)(i * (l + 2) % nk) / nk;
        });
}

// computation kernel
void kernel_2mm(num_t alpha, num_t beta, auto tmp, auto A, auto B, auto C, auto D) {
    // tmp: i x j
    // A: i x k
    // B: k x j
    // C: j x l
    // D: i x l

    noarr::traverser(tmp, A, B)
        .template for_dims<'i', 'j'>([=](auto inner) {
            auto state = inner.state();

            tmp[state] = 0;

            inner.template for_each<'k'>([=](auto state) {
                tmp[state] += alpha * A[state] * B[state];
            });
        });

    noarr::traverser(C, D, tmp)
        .template for_dims<'i', 'l'>([=](auto inner) {
            auto state = inner.state();

            D[state] *= beta;

            inner.template for_each<'j'>([=](auto state) {
                D[state] += tmp[state] * C[state];
            });
        });
}

} // namespace

int main() { /* placeholder */}
