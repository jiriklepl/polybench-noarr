#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_trmm(num_t alpha, num_t beta, auto A, auto B) {
    // A: k x i
    // B: i x j

    auto B_renamed = B ^ noarr::rename<'i', 'k'>();

    noarr::traverser(A, B)
        .template for_dims<'i', 'j'>([=](auto inner) {
            auto state = inner.state();

            inner
                .template for_each<'k'>([=](auto state) {
                    B[state] += A[state] * B_renamed[state];
                });
            
            B[state] *= alpha;
        });
}

} // namespace
