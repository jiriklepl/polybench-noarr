#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_fdtd_2d(std::size_t max, auto ex, auto ey, auto hz, auto _fict_) {
    noarr::traverser(ex, ey, hz, _fict_)
        .template for_dims<'t'>([=](auto inner) {
            auto state = inner.state();

            inner
                .order(noarr::shift<'i'>(1))
                .template for_each<'j'>([=](auto state) {
                    ey[state & noarr::idx<'i'>(0)] = _fict_[state];
                });

            inner
                .order(noarr::shift<'j'>(1))
                .template for_each([=](auto state) {
                    ey[state] = ey[state] - (num_t).5 * (hz[state] - hz[noarr::neighbor<'i'>(state, -1)]);
                });
            
            inner
                .order(noarr::span<'i'>(0, noarr::get_length<'i'>(inner) - 1)
                     ^ noarr::span<'j'>(0, noarr::get_length<'j'>(inner) - 1))
                .template for_each([=](auto state) {
                    hz[state] = hz[state] - (num_t).7 * (
                        ex[noarr::neighbor<'j'>(state, +1)] -
                        ex[state] +
                        ey[noarr::neighbor<'i'>(state, +1)] -
                        ey[state]);
                });
        });
}

} // namespace
