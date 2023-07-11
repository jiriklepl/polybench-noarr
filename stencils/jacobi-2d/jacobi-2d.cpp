#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/shortcuts.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

// initialization function
void init_array(auto A, auto B) {
    // A: i x j
    // B: i x j

    auto n = A | noarr::get_length<'i'>();

    noarr::traverser(A, B)
        .for_each([=](auto state) {
            auto [i, j] = state | noarr::get_indices<'i', 'j'>(state);

            A[state] = ((num_t)i * (j + 2) + 2) / n;
            B[state] = ((num_t)i * (j + 3) + 3) / n;
        });
}


// computation kernel
void kernel_jacobi_2d(std::size_t steps, auto A, auto B) {
    // A: i x j
    // B: i x j

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

int main() { /* placeholder */}
