#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

// initialization function
void init_array(auto L, auto x, auto b) {
    // L: i x j
    // x: i
    // b: i

    auto n = L | noarr::get_length<'i'>();

    noarr::traverser(L, x, b)
        .template for_dims<'i'>([=](auto inner) {
            auto state = inner.state();
            auto i = noarr::get_index<'i'>(state);

            x[state] = -999;
            b[state] = i;

            inner
                .order(noarr::slice<'j'>(0, i + 1))
                .template for_each<'j'>([=](auto state) {
                    auto j = noarr::get_index<'j'>(state);
                    L[state] = (num_t)(i + n - j + 1) * 2 / n;
                });
        });
}

// computation kernel
void kernel_trisolv(auto L, auto x, auto b) {
    // L: i x j
    // x: i
    // b: i

    auto x_j = x ^ noarr::rename<'i', 'j'>();

    noarr::traverser(L, x, b)
        .template for_dims<'i'>([=](auto inner) {
            auto state = inner.state();

            x[state] = b[state];

            inner
                .order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state)))
                .template for_each<'j'>([=](auto state) {
                    b[state] -= L[state] * x_j[state];
                });

            x[state] /= L[state ^ noarr::fix<'j'>(noarr::get_index<'i'>(state))];
        });
}

} // namespace

int main() { /* placeholder */}
