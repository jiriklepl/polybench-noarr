#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_bicg(auto A, auto s, auto q, auto p, auto r) {
    noarr::traverser(s)
        .for_each([=](auto state) {
            s[state] = 0;
        });
    
    noarr::traverser(A, s, q, p, r)
        .template for_dims<'i'>([=](auto inner) {
            auto state = inner.state();

            q[state] = 0;

            inner.template for_each<'j'>([=](auto state) {
                s[state] += A[state] * r[state];
                q[state] += A[state] * p[state];
            });
        });
}

} // namespace
