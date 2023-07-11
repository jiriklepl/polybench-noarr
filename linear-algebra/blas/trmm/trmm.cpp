#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

// initialization function
void init_array(num_t &alpha, auto A, auto B) {
    alpha = 1.5;

    auto ni = A | noarr::get_length<'i'>();
    auto nj = A | noarr::get_length<'j'>();

    noarr::traverser(A)
        .template for_dims<'k'>([=](auto inner) {
            auto state = inner.state();

            auto k = noarr::get_index<'k'>(state);

            inner.order(noarr::slice<'i'>(0, k))
                .template for_each<'i'>([=](auto state) {
                    auto i = noarr::get_index<'i'>(state);
                    A[state] = (num_t)((k + i) % ni) / ni;
                });
            
            A[state & noarr::idx<'i'>(k)] = 1.0;
        });

    noarr::traverser(B)
        .template for_each([=](auto state) {

            auto [i, j] = noarr::get_indices<'i', 'j'>(state);

            B[state] = (num_t)((nj + (i - j)) % nj) / nj;
        });
}

// computation kernel
void kernel_trmm(num_t alpha, auto A, auto B) {
    // A: k x i
    // B: i x j

    auto B_renamed = B ^ noarr::rename<'i', 'k'>();

    noarr::traverser(A, B)
        .template for_dims<'i', 'j'>([=](auto inner) {
            auto state = inner.state();

            inner
                .template for_each<'k'>([=](auto state) {
                    B[state] += A[state] * B_renamed[state];
                });
            
            B[state] *= alpha;
        });
}

} // namespace

int main() { /* placeholder */}
