#include "noarr/structures/extra/shortcuts.hpp"
#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_symm(num_t alpha, num_t beta, auto C, auto A, auto B) {
    auto C_renamed = C ^ noarr::rename<'i', 'k'>();
    auto B_renamed = B ^ noarr::rename<'i', 'k'>();

    noarr::traverser(C, A, B)
        .template for_dims<'i', 'j'>([=](auto inner) {
            num_t temp = 0;
            auto state = inner.state();

            inner
                .order(noarr::slice<'k'>(0, noarr::get_index<'i'>(state)))
                . template for_each<'k'>([=, &temp](auto state) {
                    C_renamed[state] += alpha * A[state] * B[state];
                    temp += A[state] * B_renamed[state];
                });

            C[state] = beta * C[state] + alpha * B[state] * A[state] + alpha * temp;
        });
}

} // namespace
