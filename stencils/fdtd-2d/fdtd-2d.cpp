#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures/extra/shortcuts.hpp>
#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "fdtd-2d.hpp"

using num_t = DATA_TYPE;

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
                .order(noarr::span<'i'>(0, (inner.top_struct() | noarr::get_length<'i'>()) - 1)
                     ^ noarr::span<'j'>(0, (inner.top_struct() | noarr::get_length<'j'>()) - 1))
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

int main(int argc, char *argv[]) {
    using namespace std::string_literals;

    // problem size
    std::size_t t = TMAX;
    std::size_t ni = NI;
    std::size_t nj = NJ;

    // data
    auto ex = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(ni, nj));
    auto ey = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(ni, nj));
    auto hz = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(ni, nj));
    auto _fict_ = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'t'>(t));

    // initialize data
    init_array(ex.get_ref(), ey.get_ref(), hz.get_ref(), _fict_.get_ref());

    auto start = std::chrono::high_resolution_clock::now();

    // run kernel
    kernel_fdtd_2d(ex.get_ref(), ey.get_ref(), hz.get_ref(), _fict_.get_ref());

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // print results
    if (argv[0] != ""s) {
        std::cout << std::fixed << std::setprecision(2);
        noarr::serialize_data(std::cout, ex.get_ref() ^ noarr::hoist<'i'>());
        noarr::serialize_data(std::cout, ey.get_ref() ^ noarr::hoist<'i'>());
        noarr::serialize_data(std::cout, hz.get_ref() ^ noarr::hoist<'i'>());
    }

    std::cerr << duration << std::endl;
}
