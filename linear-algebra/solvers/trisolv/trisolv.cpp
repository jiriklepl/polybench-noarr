#include <chrono>
#include <cstddef>
#include <iostream>

#include <noarr/structures/extra/shortcuts.hpp>
#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

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

int main(int argc, char *argv[]) {
    using namespace std::string_literals;

    // problem size
    std::size_t n = N;

    // data
    auto L = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(n, n));
    auto x = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));
    auto b = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));

    // initialize data
    init_array(L.get_ref(), x.get_ref(), b.get_ref());

    auto start = std::chrono::high_resolution_clock::now();

    // run kernel
    kernel_trisolv(L.get_ref(), x.get_ref(), b.get_ref());

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // print results
    if (argv[0] != ""s)
        noarr::serialize_data(std::cout, x);

    std::cout << duration.count() << std::endl;
}
