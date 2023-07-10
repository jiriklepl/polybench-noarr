#include "noarr/structures/extra/shortcuts.hpp"
#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_gemver(num_t alpha, num_t beta, auto A, auto u1, auto u2, auto v1, auto v2, auto w, auto x, auto y, auto z) {
    noarr::traverser(A, u1, u2, v1, v2)
        .for_each([=](auto state) {
            A[state] = A[state] + u1[state] * v1[state] + u2[state] * v2[state];
        });

    noarr::traverser(x, A, y)
        .for_each([=](auto state) {
            x[state] = x[state] + beta * A[state] * y[state];
        });
    
    noarr::traverser(x, z)
        .for_each([=](auto state) {
            x[state] = x[state] + z[state];
        });

    noarr::traverser(A, w, x)
        .for_each([=](auto state) {
           w[state] = w[state] + alpha * A[state] * x[state];
        });
}

} // namespace
