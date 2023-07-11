#include <chrono>
#include <iostream>

#include <noarr/structures/extra/shortcuts.hpp>
#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "covariance.hpp"

using num_t = DATA_TYPE;

namespace {

void init_array(num_t &float_n, auto data) {
    // data: k x j

    float_n = data | noarr::get_length<'k'>();

    noarr::traverser(data)
        .template for_each([=](auto state) {
            auto [k, j] = noarr::get_indices<'k', 'j'>(state);
            data[state] = (num_t)(k * j) / (data | noarr::get_length<'j'>());
        });
}


void kernel_covariance(num_t float_n, auto data, auto cov, auto mean) {
    // data: k x j
    // cov: i x j
    // mean: j

    auto cov_ji = cov ^ noarr::rename<'i', 'j', 'j', 'i'>();
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

    noarr::traverser(data, mean)
        .template for_each<'k', 'j'>([=](auto state) {
            data[state] -= mean[state];
        });

    noarr::traverser(data, cov, mean)
        .template for_dims<'i'>([=](auto inner) {
            inner
                .order(noarr::shift<'j'>(noarr::get_index<'i'>(inner.state())))
                .template for_dims<'j'>([=](auto inner) {
                    auto state = inner.state();

                    cov[state] = 0;

                    inner.template for_each<'k'>([=](auto state) {
                        cov[state] += data[state] * data_ki[state];
                    });

                    cov[state] /= float_n - (num_t)1;
                    cov_ji[state] = cov[state];
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
    auto cov = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(nj, nj));
    auto mean = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'j'>(nj));

    // initialize data
    init_array(float_n, data.get_ref());

    auto start = std::chrono::high_resolution_clock::now();

    // run kernel
    kernel_covariance(float_n, data.get_ref(), cov.get_ref(), mean.get_ref());

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // print results
    if (argv[0] != ""s)
        noarr::serialize_data(std::cout, cov.get_ref());

    std::cout << duration.count() << std::endl;
}
