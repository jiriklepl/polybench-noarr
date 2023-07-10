#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_2mm(num_t alpha, num_t beta, auto tmp, auto A, auto B, auto C, auto D) {
    noarr::traverser(tmp, A, B)
        .template for_dims<'i', 'j'>([=](auto inner) {
            auto state = inner.state();

            tmp[state] = 0;

            inner.template for_each<'k'>([=](auto state) {
                tmp[state] += alpha * A[state] * B[state];
            });
        });

    noarr::traverser(C, D, tmp)
        .template for_dims<'i', 'j'>([=](auto inner) {
            auto state = inner.state();

            D[state] *= beta;

            inner.template for_each<'k'>([=](auto state) {
                D[state] += tmp[state] * C[state];
            });
        });
}

} // namespace
