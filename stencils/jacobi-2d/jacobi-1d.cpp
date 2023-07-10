#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/shortcuts.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_jacobi_2d(std::size_t steps, auto A, auto B) {
    auto traverser = noarr::traverser(A, B).order(noarr::bcast<'t'>(steps));

    traverser
        .order(noarr::symmetric_spans<'i', 'j'>(traverser.top_struct(), 1, 1))
        .template for_dims<'t'>([=](auto inner) {
            inner.for_each([=](auto state) {
                B[state] = .2 * (
                    A[neighbor<'i'>(state, -1)] +
                    A[neighbor<'i'>(state, +1)] +
                    A[neighbor<'j'>(state, -1)] +
                    A[neighbor<'j'>(state, +1)] +
                    A[state]);
            });

            inner.for_each([=](auto state) {
                A[state] = .2 * (
                    B[neighbor<'i'>(state, -1)] +
                    B[neighbor<'i'>(state, +1)] +
                    B[neighbor<'j'>(state, -1)] +
                    B[neighbor<'j'>(state, +1)] +
                    B[state]);
            });
        });
}

} // namespace
