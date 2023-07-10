#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_atax(auto A, auto x, auto y, auto tmp) {
    noarr::traverser(y)
        .for_each([=](auto state) {
            y[state] = 0;
        });
    
    noarr::traverser(A, x, y, tmp)
        .template for_dims<'i'>([=](auto inner) {
            auto state = inner.state();

            tmp[state] = 0;

            inner.template for_each<'j'>([=](auto state) {
                tmp[state] += A[state] * x[state];
            });

            inner.template for_each<'j'>([=](auto state) {
                y[state] += A[state] * tmp[state];
            });
        });
}

} // namespace
