#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

// initialization function
void init_array(auto ex, auto ey, auto hz, auto _fict_) {
    // ex: i x j
    // ey: i x j
    // hz: i x j
    // _fict_: t

    auto ni = ex | noarr::get_length<'i'>();
    auto nj = ex | noarr::get_length<'j'>();
    
    noarr::traverser(_fict_)
        .template for_each([=](auto state) {
            auto t = noarr::get_index<'t'>(state);
            _fict_[state] = t;
        });

    noarr::traverser(ex, ey, hz)
        .template for_each([=](auto state) {
            auto [i, j] = noarr::get_indices<'i', 'j'>(state);

            ex[state] = ((num_t) i * (j + 1)) / ni;
            ey[state] = ((num_t) i * (j + 2)) / nj;
            hz[state] = ((num_t) i * (j + 3)) / ni;
        });
}


// computation kernel
void kernel_fdtd_2d(auto ex, auto ey, auto hz, auto _fict_) {
    // ex: i x j
    // ey: i x j
    // hz: i x j
    // _fict_: t

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

int main() { /* placeholder */}
