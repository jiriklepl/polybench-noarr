#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

// initialization function
void init_array(auto A, auto C4) {
    // A: r x q x p
    // C4: s x p

    auto np = A | noarr::get_length<'p'>();

    noarr::traverser(A)
        .for_each([=](auto state) {
            auto [r, q, p] = noarr::get_indices<'r', 'q', 'p'>(state);
            A[state] = (num_t)((r * q + p) % np) / np;
        });

    noarr::traverser(C4)
        .for_each([=](auto state) {
            auto [s, p] = noarr::get_indices<'s', 'p'>(state);
            C4[state] = (num_t)(s * p % np) / np;
        });
}

// computation kernel
void kernel_doitgen(auto A, auto C4, auto sum) {
    // A: r x q x p
    // C4: s x p
    // sum: p

    auto A_rqs = A | noarr::rename<'p', 's'>();

    noarr::traverser(A, C4, sum)
        .template for_dims<'r', 'q'>([=](auto inner) {
            inner.template for_dims<'p'>([=](auto inner) {
                auto state = inner.state();

                sum[state] = 0;

                inner.template for_each<'s'>([=](auto state) {
                    sum[state] += A_rqs[state] * C4[state];
                });
            });

            inner.template for_each<'p'>([=](auto state) {
                A[state] = sum[state];
            });
        });
}

} // namespace

int main() { /* placeholder */}
