#include "noarr/structures/extra/shortcuts.hpp"
#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_gesummv(num_t alpha, num_t beta, auto A, auto B, auto tmp, auto x, auto y) {
    noarr::traverser(A, B, tmp, x, y)
        .template for_dims<'i'>([=](auto inner) {
            auto state = inner.state();

            tmp[state] = 0;
            y[state] = 0;

            inner.template for_each<'j'>([=](auto state) {
                tmp[state] += A[state] * x[state];
                y[state] += B[state] * x[state];
            });

            y[state] = alpha * tmp[state] + beta * y[state];
        });
}

} // namespace
