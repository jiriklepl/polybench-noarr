#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_mvt(auto x1, auto x2, auto y1, auto y2, auto A) {
    noarr::traverser(x1, A, y1)
        .template for_each<'i', 'j'>([=](auto state) {
            x1[state] += A[state] * y1[state];
        });

    noarr::traverser(x2, A, y2)
        .template for_each<'i', 'j'>([=](auto state) {
            x2[state] += A[state] * y2[state];
        });
}

} // namespace
