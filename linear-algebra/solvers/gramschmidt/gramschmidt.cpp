#include <cmath>

#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

template<class V, class U>
float inner_product(V v, U u) {
    float result = 0;

    noarr::traverser(u, v).for_each([&result, u, v](auto state) {
        result += u[state] * v[state];
    });

    return result;
}

void kernel_gramschmidt(auto A, auto R, auto Q) {
    // A: i x k
    // R: k x j
    // Q: i x k
    auto A_ij = A ^ noarr::rename<'k', 'j'>();

    noarr::traverser(A_ij, R, Q)
        .template for_dims<'k'>([=](auto inner) {
            auto state = inner.state();
            num_t norm = 0;

            inner.template for_each<'i'>([=](auto state) {
                norm += A[state] * A[state];
            });

            auto R_diag = R ^ noarr::fix<'j'>(noarr::get_index<'k'>(state));

            R[state] = std::sqrt(norm);

            inner.template for_each<'i'>([=](auto state) {
                Q[state] = A[state] / R[state];
            });

            inner
                .order(noarr::shift<'j'>(noarr::get_index<'k'>(state) + 1))
                .template for_dims<'j'>([=](auto inner) {
                    auto state = inner.state();

                    R[state] = 0;

                    inner.template for_each<'i'>([=](auto state) {
                        R[state] += Q[state] * A_ij[state];
                    });

                    inner.template for_each<'i'>([=](auto state) {
                        A_ij[state] -= Q[state] * R[state];
                    });
                });                     
        });
}

} // namespace
