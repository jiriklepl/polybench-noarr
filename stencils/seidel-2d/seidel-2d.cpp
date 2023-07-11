#include "noarr/structures/structs/views.hpp"
#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

// initialization function
void init_array(auto A) {
    // A: i x j

    auto n = noarr::get_length<'i'>(A);

    noarr::traverser(A)
        .for_each([=](auto state) {
            auto [i, j] = state | noarr::get_indices<'i', 'j'>(state);

            A[state] = ((num_t)i * (j + 2) + 2) / n;
        });
}

// computation kernel
void kernel_seidel_2d(std::size_t steps, auto A) {
    // A: i x j

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

int main() { /* placeholder */}
