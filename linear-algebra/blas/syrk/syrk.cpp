#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_syrk(num_t alpha, num_t beta, auto C, auto A) {
    auto A_renamed = A ^ noarr::rename<'i', 'j'>();

    noarr::traverser(C, A)
        .template for_dims<'i'>(
            [=](auto inner) {
                auto state = inner.state();

                inner
                    .order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state)))
                    .template for_each<'j'>([=](auto state) {
                        C[state] *= beta;
                    });

                inner
                    .order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state)))
                    .template for_each<'k', 'j'>([=](auto state) {
                        C[state] += alpha * A_renamed[state] * A[state];
                    });
            });
}

} // namespace
