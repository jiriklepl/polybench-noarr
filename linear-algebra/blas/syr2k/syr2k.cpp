#include "noarr/structures/extra/shortcuts.hpp"
#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

// initialization function
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) {
    // C: i x j
    // A: i x k
    // B: i x k

    alpha = 1.5;
    beta = 1.2;

    auto ni = C | noarr::get_length<'i'>();
    auto nk = A | noarr::get_length<'k'>();

    noarr::traverser(A, B)
        .for_each([=](auto state) {
            auto [i, k] = noarr::get_indices<'i', 'k'>(state);
            A[state] = (num_t)((i * k + 1) % ni) / ni;
            B[state] = (num_t)((k * i + 2) % nk) / nk;
        });

    noarr::traverser(C)
        .for_each([=](auto state) {
            auto [i, j] = noarr::get_indices<'i', 'j'>(state);
            C[state] = (num_t)((i * j + 3) % ni) / nk;
        });
}

// computation kernel
void kernel_syr2k(num_t alpha, num_t beta, auto C, auto A, auto B) {
    // C: i x j
    // A: i x k
    // B: i x k

    auto A_renamed = A ^ noarr::rename<'i', 'j'>();
    auto B_renamed = B ^ noarr::rename<'i', 'j'>();

    noarr::traverser(C, A, B)
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
                        C[state] += alpha * A_renamed[state] * B[state] + alpha * B_renamed[state] * A[state];
                    });
            });
}

} // namespace

int main() { /* placeholder */}
