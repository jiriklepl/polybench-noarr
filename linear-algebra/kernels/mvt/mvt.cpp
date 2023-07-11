#include <chrono>
#include <iostream>

#include <noarr/structures/extra/shortcuts.hpp>
#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "mvt.hpp"

using num_t = DATA_TYPE;

namespace {

// initialization function
void init_array(auto x1, auto x2, auto y1, auto y2, auto A) {
    // x1: i
    // x2: i
    // y1: j
    // y2: j
    // A: i x j

    auto n = A | noarr::get_length<'i'>();

    auto y1_i = y1 ^ noarr::rename<'j', 'i'>();
    auto y2_i = y2 ^ noarr::rename<'j', 'i'>();

    noarr::traverser(x1, x2, y1_i, y2_i, A)
        .template for_dims<'i'>([=](auto inner) {
            auto state = inner.state();
            auto i = noarr::get_index<'i'>(state);
            
            x1[state] = (num_t)(i % n) / n;
            x2[state] = (num_t)((i + 1) % n) / n;
            y1_i[state] = (num_t)((i + 3) % n) / n;
            y2_i[state] = (num_t)((i + 4) % n) / n;

            inner.template for_each<'j'>([=](auto state) {
                auto j = noarr::get_index<'j'>(state);

                A[state] = (num_t)(i * j % n) / n;
            });
        });
}

// computation kernel
void kernel_mvt(auto x1, auto x2, auto y1, auto y2, auto A) {
    // x1: i
    // x2: i
    // y1: j
    // y2: j
    // A: i x j
    auto A_ji = A ^ noarr::rename<'i', 'j', 'j', 'i'>();

    noarr::traverser(x1, A, y1)
        .template for_each<'i', 'j'>([=](auto state) {
            x1[state] += A[state] * y1[state];
        });

    noarr::traverser(x2, A, y2)
        .template for_each<'i', 'j'>([=](auto state) {
            x2[state] += A_ji[state] * y2[state];
        });
}

} // namespace

int main(int argc, char *argv[]) {
    using namespace std::string_literals;

    // problem size
    std::size_t n = N;

    // data
    auto x1 = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));
    auto x2 = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));

    auto y1 = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'j'>(n));
    auto y2 = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'j'>(n));

    auto A = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(n, n));

    // initialize data
    init_array(x1.get_ref(), x2.get_ref(), y1.get_ref(), y2.get_ref(), A.get_ref());

    auto start = std::chrono::high_resolution_clock::now();

    // run kernel
    kernel_mvt(x1.get_ref(), x2.get_ref(), y1.get_ref(), y2.get_ref(), A.get_ref());

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // print results
    if (argv[0] != ""s) {
        noarr::serialize_data(std::cout, x1);
        noarr::serialize_data(std::cout, x2);
    }

    std::cerr << duration << std::endl;
}
