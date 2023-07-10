#include "noarr/structures/extra/shortcuts.hpp"
#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_cholesky(auto A) {
    auto A_ik = A ^ noarr::rename<'j', 'k'>();
    auto A_jk = A ^ noarr::rename<'i', 'j', 'j', 'k'>();

    noarr::traverser(A, A_ik)
        .template for_dims<'i'>([=](auto inner) {
            auto state = inner.state();

            inner
                .order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state)))
                .for_each([=](auto state) {
                    A[state] -= A_ik[state] * A_jk[state];
                });

            auto A_ii = A ^ noarr::fix<'j'>(noarr::get_index<'i'>(state));

            inner
                .template for_each<'k'>([=](auto state) {
                    A_ii[state] -= A_ik[state] * A_ik[state];
                });

            A_ii[state] = sqrt(A_ii[state]);
        });
}

} // namespace
