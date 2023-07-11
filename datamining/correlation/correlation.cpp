#include <cmath>
#include <chrono>
#include <iostream>

#include <noarr/structures/extra/shortcuts.hpp>
#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "correlation.hpp"

using num_t = DATA_TYPE;

namespace {

void init_array(num_t &float_n, auto data) {
    // data: k x j

    float_n = data | noarr::get_length<'k'>();

    noarr::traverser(data)
        .template for_each([=](auto state) {
            auto [k, j] = noarr::get_indices<'k', 'j'>(state);
            data[state] = (num_t)(k * j) / (data | noarr::get_length<'j'>()) + k;
        });
}

void kernel_correlation(num_t float_n, auto data, auto corr, auto mean, auto stddev) {
    // data: k x j
    // corr: i x j
    // mean: j
    // stddev: j

    num_t eps = .1;

    auto corr_ji = corr ^ noarr::rename<'i', 'j', 'j', 'i'>();
    auto data_ki = data ^ noarr::rename<'j', 'i'>();

    noarr::traverser(data, mean)
        .template for_dims<'j'>([=](auto inner) {
            auto state = inner.state();

            mean[state] = 0;

            inner.template for_each<'k'>([=](auto state) {
                mean[state] += data[state];
            });

            mean[state] /= float_n;
        });

    noarr::traverser(data, mean, stddev)
        .template for_dims<'j'>([=](auto inner) {
            auto state = inner.state();

            stddev[state] = 0;

            inner.template for_each<'k'>([=](auto state) {
                stddev[state] += (data[state] - mean[state]) * (data[state] - mean[state]);
            });

            stddev[state] /= float_n; // TODO?: - 1
            stddev[state] = std::sqrt(stddev[state]);
            stddev[state] = stddev[state] <= eps ? (num_t)1 : stddev[state];
        });

    noarr::traverser(data, mean, stddev)
        .template for_each<'k', 'j'>([=](auto state) {
            data[state] -= mean[state];
            data[state] /= std::sqrt(float_n) * stddev[state];
        });

    auto traverser = noarr::traverser(data, corr, data_ki, corr_ji);
    traverser
        .template for_dims<'i'>([=](auto inner) {
            auto state = inner.state();

            corr[state & noarr::idx<'j'>(noarr::get_index<'i'>(state))] = 1; // TODO: corr_diag

            inner
                .order(noarr::shift<'j'>(noarr::get_index<'i'>(state) + 1))
                .template for_dims<'j'>([=](auto inner) {
                    auto state = inner.state();

                    corr[state] = 0;

                    inner.template for_each<'k'>([=](auto state) {
                        corr[state] += data_ki[state] * data[state];
                    });

                    corr_ji[state] = corr[state];
                    corr_ji[state] /= float_n;
                });
        });
}

} // namespace

int main(int argc, char *argv[]) {
    using namespace std::string_literals;

    // problem size
    std::size_t nk = NK;
    std::size_t nj = NJ;

    // data
    num_t float_n;
    auto data = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'k', 'j'>(nk, nj));
    auto corr = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(nj, nj));
    auto mean = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'j'>(nj));
    auto stddev = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'j'>(nj));

    // initialize data
    init_array(float_n, data.get_ref());

    auto start = std::chrono::high_resolution_clock::now();

    // run kernel
    kernel_correlation(float_n, data.get_ref(), corr.get_ref(), mean.get_ref(), stddev.get_ref());

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // print results
    if (argv[0] != ""s)
        noarr::serialize_data(std::cout, corr.get_ref());

    std::cerr << duration << std::endl;
}
