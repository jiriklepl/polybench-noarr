#include "noarr/structures/extra/shortcuts.hpp"
#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

// initialization function
void init_array(num_t &alpha, num_t &beta, auto A, auto B, auto x) {
    alpha = 1.5;
    beta = 1.2;

    auto n = x | noarr::get_length<'i'>();

    noarr::traverser(A, B, x)
        .template for_dims<'i'>([=](auto inner) {
            auto state = inner.state();

            auto i = noarr::get_index<'i'>(state);

            x[state] = (num_t)(i % n) / n;

            inner.template for_each<'j'>([=](auto state) {
                auto j = noarr::get_index<'j'>(state);

                A[state] = (num_t)((i * j + 1) % n) / n;
                B[state] = (num_t)((i * j + 2) % n) / n;
            });
        });
}

// computation kernel
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

int main() { /* placeholder */}
