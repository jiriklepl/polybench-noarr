#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/shortcuts.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_jacobi_1d(std::size_t steps, auto A, auto B) {
    auto traverser = noarr::traverser(A, B).order(noarr::bcast<'t'>(steps));

    traverser
        .order(noarr::symmetric_span<'i'>(traverser.top_struct(), 1))
        .template for_dims<'t'>([=](auto inner) {
            inner.for_each([=](auto state) {
                B[state] = 0.333 * (A[neighbor<'i'>(state, -1)] + A[state] + A[neighbor<'i'>(state, +1)]);
            });

            inner.for_each([=](auto state) {
                A[state] = 0.333 * (B[neighbor<'i'>(state, -1)] + B[state] + B[neighbor<'i'>(state, +1)]);
            });
        });
}

} // namespace
