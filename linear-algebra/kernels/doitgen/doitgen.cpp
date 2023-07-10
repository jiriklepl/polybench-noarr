#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_doitgen(auto A, auto C4, auto sum) {
    noarr::traverser(A, C4, sum)
        .template for_dims<'r', 'q'>([=](auto inner) {
            inner.template for_dims<'p'>([=](auto inner) {
                auto state = inner.state();

                sum[state] = 0;

                inner.template for_each<'s'>([=](auto state) {
                    sum[state] += A[state] * C4[state];
                });
            });

            inner.template for_each<'p'>([=](auto state) {
                A[state] = sum[state];
            });
        });
}

} // namespace
