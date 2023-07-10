#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_heat_3d(std::size_t steps, auto A, auto B) {
    auto traverser = noarr::traverser(A, B).order(noarr::bcast<'t'>(steps));

    traverser
        .order(noarr::symmetric_spans<'i', 'j', 'k'>(traverser.top_struct(), 1, 1, 1))
        .template for_dims<'t'>([=](auto inner) {
            inner.for_each([=](auto state) {
                B[state] =
                    (num_t).125 * (A[neighbor<'i'>(state, -1)] -
                                   2 * A[state] +
                                   A[neighbor<'i'>(state, +1)]) +
                    (num_t).125 * (A[neighbor<'j'>(state, -1)] -
                                   2 * A[state] +
                                   A[neighbor<'j'>(state, +1)]) +
                    (num_t).125 * (A[neighbor<'k'>(state, -1)] -
                                   2 * A[state] +
                                   A[neighbor<'k'>(state, +1)]) +
                    A[state];
            });

            inner.for_each([=](auto state) {
                A[state] =
                    (num_t).125 * (B[neighbor<'i'>(state, -1)] -
                                   2 * B[state] +
                                   B[neighbor<'i'>(state, +1)]) +
                    (num_t).125 * (B[neighbor<'j'>(state, -1)] -
                                   2 * B[state] +
                                   B[neighbor<'j'>(state, +1)]) +
                    (num_t).125 * (B[neighbor<'k'>(state, -1)] -
                                   2 * B[state] +
                                   B[neighbor<'k'>(state, +1)]) +
                    B[state];
            });
        });
}

} // namespace
