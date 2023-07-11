#include "noarr/structures/extra/shortcuts.hpp"
#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

// initialization function
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) {
    alpha = 1.5;
    beta = 1.2;

    auto ni = C | noarr::get_length<'i'>();
    auto nj = C | noarr::get_length<'j'>();
    auto nk = A | noarr::get_length<'k'>();

    noarr::traverser(C)
        .for_each([=](auto state) {
            auto [i, j] = noarr::get_indices<'i', 'j'>(state);
            C[state] = (num_t)((i + j) % 100) / ni;
            B[state] = (num_t)((nj + i - j) % 100) / ni;
        });

    noarr::traverser(A)
        .template for_dims<'i'>([=](auto inner) {
            auto state = inner.state();
            
            inner.order(noarr::slice<'k'>(0, noarr::get_index<'i'>(state) + 1))
                .for_each([=](auto state) {
                    auto [i, k] = noarr::get_indices<'i', 'k'>(state);
                    A[state] = (num_t)((i + k) % 100) / ni;
                });
            
            inner.order(noarr::shift<'k'>(noarr::get_index<'i'>(state) + 1))
                .for_each([=](auto state) {
                    auto [i, k] = noarr::get_indices<'i', 'k'>(state);
                    A[state] = -999;
                });
        });
}

// computation kernel
void kernel_symm(num_t alpha, num_t beta, auto C, auto A, auto B) {
    auto C_renamed = C ^ noarr::rename<'i', 'k'>();
    auto B_renamed = B ^ noarr::rename<'i', 'k'>();

    noarr::traverser(C, A, B)
        .template for_dims<'i', 'j'>([=](auto inner) {
            num_t temp = 0;
            auto state = inner.state();

            inner
                .order(noarr::slice<'k'>(0, noarr::get_index<'i'>(state)))
                . template for_each<'k'>([=, &temp](auto state) {
                    C_renamed[state] += alpha * A[state] * B[state];
                    temp += A[state] * B_renamed[state];
                });

            C[state] = beta * C[state] + alpha * B[state] * A[state] + alpha * temp;
        });
}

} // namespace

int main() { /* placeholder */}
