#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

// initialization function
void init_array(num_t &alpha, num_t &beta, auto C, auto A) {
    alpha = 1.5;
    beta = 1.2;

    auto ni = C | noarr::get_length<'i'>();
    auto nk = A | noarr::get_length<'k'>();

    noarr::traverser(A)
        .for_each([=](auto state) {
            auto [i, k] = noarr::get_indices<'i', 'k'>(state);

            A[state] = (num_t)((i * k + 1) % ni) / ni;
        });

    noarr::traverser(C)
        .for_each([=](auto state) {
            auto [i, j] = noarr::get_indices<'i', 'j'>(state);

            C[state] = (num_t)((i * j + 2) % nk) / nk;
        });
}

// computation kernel
void kernel_syrk(num_t alpha, num_t beta, auto C, auto A) {
    // C: i x j
    // A: i x k
    auto A_renamed = A ^ noarr::rename<'i', 'j'>();

    noarr::traverser(C, A)
        .template for_dims<'i'>(
            [=](auto inner) {
                auto state = inner.state();

                inner
                    .order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state)))
                    .template for_each<'j'>([=](auto state) {
                        C[state] *= beta;
                    });

                inner
                    .order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state)))
                    .template for_each<'k', 'j'>([=](auto state) {
                        C[state] += alpha * A_renamed[state] * A[state];
                    });
            });
}

} // namespace

int main() { /* placeholder */}
