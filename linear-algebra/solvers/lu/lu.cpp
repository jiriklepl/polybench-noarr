#include "noarr/structures/extra/shortcuts.hpp"
#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void init_array(auto A) {
    auto n = noarr::get_length<'i'>(A);

    noarr::traverser(A)
        .template for_dims<'i'>([=](auto inner) {
            auto state = inner.state();

            inner
                .order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state)))
                .template for_each<'j'>([=](auto state) {
                    A[state] = (num_t) (-noarr::get_index<'j'>(state) % n) / n + 1;
                });
            
            inner
                .order(noarr::shift<'j'>(0, noarr::get_index<'i'>(state) + 1))
                .template for_each<'j'>([=](auto state) {
                    A[state] = 0;
                });
            
            A[state & noarr::idx<'j'>(noarr::get_index<'i'>(state))] = 1;
        });
}

void kernel_lu(auto A) {
    // A: i x j
    auto A_ik = A ^ noarr::rename<'j', 'k'>();
    auto A_kj = A ^ noarr::rename<'i', 'k'>();

    noarr::traverser(A, A_ik, A_kj)
        .template for_dims<'i'>([=](auto inner) {
            auto state = inner.state();

            inner
                .order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state)))
                .template for_dims<'j'>([=](auto inner) {
                    auto state = inner.state();
                    
                    inner
                        .order(noarr::slice<'k'>(0, noarr::get_index<'j'>(state)))
                        .template for_each<'k'>([=](auto state) {
                            A[state] -= A_ik[state] * A_kj[state];
                        });

                    A[state] /= (A ^ fix<'i'>(noarr::get_index<'j'>(state)))[state];
                });

            inner
                .order(noarr::shift<'j'>(0, noarr::get_index<'i'>(state)))
                .template for_each<'j', 'k'>([=](auto state) {
                    A[state] -= A_ik[state] * A_kj[state];
                });
        });
}

} // namespace
