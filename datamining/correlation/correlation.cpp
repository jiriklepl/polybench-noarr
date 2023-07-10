#include <cmath>

#include "noarr/structures/extra/shortcuts.hpp"
#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_correlation(num_t float_n, auto data, auto corr, auto mean, auto stddev) {
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
                });

            corr_ji[state] /= float_n;
        });
}

} // namespace
