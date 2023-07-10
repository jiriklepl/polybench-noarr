#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_3mm(auto E, auto A, auto B, auto F, auto C, auto D, auto G) {
    noarr::traverser(E, A, B)
        .template for_dims<'i', 'j'>([=](auto inner) {
            auto state = inner.state();

            E[state] = 0;

            inner.template for_each<'k'>([=](auto state) {
                E[state] += A[state] * B[state];
            });
        });

    noarr::traverser(F, C, D)
        .template for_dims<'i', 'j'>([=](auto inner) {
            auto state = inner.state();

            F[state] = 0;

            inner.template for_each<'k'>([=](auto state) {
                F[state] += C[state] * D[state];
            });
        });

    noarr::traverser(G, E, F)
        .template for_dims<'i', 'j'>([=](auto inner) {
            auto state = inner.state();

            G[state] = 0;

            inner.template for_each<'k'>([=](auto state) {
                G[state] += E[state] * F[state];
            });
        });
}

} // namespace
