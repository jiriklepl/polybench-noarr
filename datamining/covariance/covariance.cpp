#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void init_array(num_t &float_n, auto data) {
    float_n = data | noarr::get_length<'k'>();

    noarr::traverser(data)
        .template for_each([=](auto state) {
            auto [k, j] = noarr::get_indices<'k', 'j'>(state);
            data[state] = (num_t)(k * j) / (data | noarr::get_length<'j'>());
        });
}


void kernel_covariance(num_t float_n, auto data, auto cov, auto mean) {
    auto cov_ji = cov ^ noarr::rename<'i', 'j', 'j', 'i'>();
    auto data_ki = data ^ noarr::rename<'j', 'i'>();

    noarr::traverser(data, mean)
        .template for_dims<'k'>([=](auto inner) {
            auto state = inner.state();

            mean[state] = 0;

            inner.template for_each<'j'>([=](auto state) {
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

int main() { /* placeholder */}
