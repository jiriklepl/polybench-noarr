#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_trisolv(auto L, auto x, auto b) {
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
