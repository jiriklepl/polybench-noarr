#include "noarr/structures/structs/views.hpp"
#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_seidel_2d(std::size_t steps, auto A) {
    auto traverser = noarr::traverser(A).order(noarr::bcast<'t'>(steps));

    traverser
        .order(noarr::symmetric_spans<'i', 'j'>(traverser.top_struct(), 1, 1))
        .order(noarr::hoist<'t'>())
        .inner.for_each([=](auto state) {
            A[state] = .2 * (
                A[neighbor<'i', 'j'>(state, -1, -1)] + // corner
                A[neighbor<'i'>(state, -1)] +          // edge
                A[neighbor<'i', 'j'>(state, -1, +1)] + // corner
                A[neighbor<'j'>(state, -1)] +          // edge
                A[state] +                             // center
                A[neighbor<'j'>(state, +1)] +          // edge
                A[neighbor<'i', 'j'>(state, +1, -1)] + // corner
                A[neighbor<'i'>(state, +1)] +          // edge
                A[neighbor<'i', 'j'>(state, +1, +1)]); // corner
        });
}

} // namespace
