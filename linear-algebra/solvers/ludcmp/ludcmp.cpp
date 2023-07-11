#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_ludcmp(auto A, auto b, auto x, auto y) {
    auto A_ik = A ^ noarr::rename<'j', 'k'>();
    auto A_kj = A ^ noarr::rename<'i', 'k'>();

    noarr::traverser(A, b, A_ik, A_kj)
        .template for_dims<'i'>([=](auto inner) {
            auto state = inner.state();

            inner
                .order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state)))
                .template for_dims<'j'>([=](auto inner) {
                    auto state = inner.state();

                    num_t w = A[state];

                    inner
                        .order(noarr::slice<'k'>(0, noarr::get_index<'j'>(state)))
                        .template for_each<'k'>([=, &w](auto state) {
                            w -= A_ik[state] * A_kj[state];
                        });

                    A[state] = w / (A ^ noarr::fix<'i'>(noarr::get_index<'j'>(state)))[state];
                });

            inner
                .order(noarr::shift<'j'>(noarr::get_index<'i'>(state)))
                .template for_dims<'j'>([=](auto inner) {
                    auto state = inner.state();

                    num_t w = A[state];

                    inner
                        .order(noarr::slice<'k'>(0, noarr::get_index<'i'>(state)))
                        .template for_each<'k'>([=, &w](auto state) {
                            w -= A_ik[state] * A_kj[state];
                        });

                    A[state] = w;
                });
        });

        noarr::traverser(A, b, y)
            .template for_dims<'i'>([=](auto inner) {
                auto state = inner.state();

                num_t w = b[state];

                inner
                    .order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state)))
                    .template for_each<'j'>([=, &w](auto state) {
                        w -= A[state] * y[noarr::idx<'i'>(noarr::get_index<'j'>(state))];
                    });
                
                y[state] = w;
            });

        noarr::traverser(A, x)
            .order(noarr::reverse<'i'>())
            .template for_dims<'i'>([=](auto inner) {
                auto state = inner.state();

                num_t w = y[state];

                inner
                    .order(noarr::shift<'j'>(noarr::get_index<'i'>(state) + 1))
                    .template for_each<'j'>([=, &w](auto state) {
                        w -= A[state] * x[noarr::idx<'i'>(noarr::get_index<'j'>(state))];
                    });
                
                x[state] = w / A[state ^ noarr::fix<'j'>(noarr::get_index<'i'>(state))];
            });
}

} // namespace

int main() { /* placeholder */}
