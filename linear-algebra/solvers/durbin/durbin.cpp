#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"
#include "noarr/structures/interop/bag.hpp"

using num_t = float;

namespace {

void kernel_durbin(auto r, auto y) {
    auto z = noarr::bag(r.structure());

    auto r_k = r ^ noarr::rename<'i', 'k'>();
    auto y_k = y ^ noarr::rename<'i', 'k'>();

    num_t alpha;
    num_t beta;
    num_t sum;

    y[noarr::idx<'i'>(0)] = -r[noarr::idx<'i'>(0)];
    beta = 1;
    alpha = -r[noarr::idx<'i'>(0)];

    noarr::traverser(r, y, r_k, y_k)
        .order(noarr::shift<'k'>(1))
        .template for_dims<'k'>([=, &alpha, &beta, &sum](auto inner) {
            auto state = inner.state();

            beta = (1 - alpha * alpha) * beta;
            sum = 0;

            auto traverser = inner
                .order(noarr::slice<'i'>(0, noarr::get_index<'k'>(state)));

            traverser
                .template for_each<'i'>([=, &sum](auto state) {
                    auto [i, k] = noarr::get_indices<'i', 'k'>(state);
                    // sum += r_k[noarr::neighbor<'k'>(state, -i - 1)] * y[state];
                    sum += r[noarr::idx<'i'>(k - i - 1)] * y[state];
                });

            alpha = -(r_k[state] + sum) / beta;

            traverser
                .template for_each<'i'>([=, &alpha, &beta](auto state) {
                    auto [i, k] = noarr::get_indices<'i', 'k'>(state);
                    // z[state] = y[state] + alpha * y[noarr::neighbor<'k'>(state, -i - 1)];
                    z[state] = y[state] + alpha * y[noarr::idx<'i'>(k - i - 1)];
                });

            traverser
                .template for_each<'i'>([=, &alpha, &beta](auto state) {
                    y[state] = z[state];
                });

            y_k[state] = alpha;
        });
}

} // namespace
