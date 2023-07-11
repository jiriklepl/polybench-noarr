#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

// initialization function
void init_array(auto A, auto x) {
    // A: i x j
    // x: j

    auto ni = A | noarr::get_length<'i'>();
    auto nj = A | noarr::get_length<'j'>();

    noarr::traverser(x).for_each([=](auto state) {
        auto j = noarr::get_index<'j'>(state);
        x[state] = 1 + j / nj;
    });

    noarr::traverser(A).for_each([=](auto state) {
        auto [i, j] = noarr::get_indices<'i', 'j'>(state);
        A[state] = (num_t)((i + j) % nj) / (5 * ni);
    });
}

// computation kernel
void kernel_atax(auto A, auto x, auto y, auto tmp) {
    // A: i x j
    // x: j
    // y: j
    // tmp: i

    noarr::traverser(y)
        .for_each([=](auto state) {
            y[state] = 0;
        });
    
    noarr::traverser(A, x, y, tmp)
        .template for_dims<'i'>([=](auto inner) {
            auto state = inner.state();

            tmp[state] = 0;

            inner.template for_each<'j'>([=](auto state) {
                tmp[state] += A[state] * x[state];
            });

            inner.template for_each<'j'>([=](auto state) {
                y[state] += A[state] * tmp[state];
            });
        });
}

} // namespace

int main() { /* placeholder */}
