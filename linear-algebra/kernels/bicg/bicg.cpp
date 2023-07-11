#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

// initialization function
void init_array(auto A, auto r, auto p) {
    // A: i x j
    // r: i
    // p: j

    auto ni = A | noarr::get_length<'i'>();
    auto nj = A | noarr::get_length<'j'>();

    noarr::traverser(p).for_each([=](auto state) {
        auto j = noarr::get_index<'j'>(state);
        p[state] = (num_t)(j % nj) / nj;
    });

    noarr::traverser(A, r)
        .template for_dims<'i'>([=](auto inner) {
            auto state = inner.state();

            auto i = noarr::get_index<'i'>(state);

            r[state] = (num_t)(i % ni) / ni;

            inner.template for_each<'j'>([=](auto state) {
                auto j = noarr::get_index<'j'>(state);

                A[state] = (num_t)(i * (j + 1) % ni) / ni;
            });
        });
}

// computation kernel
void kernel_bicg(auto A, auto s, auto q, auto p, auto r) {
    // A: i x j
    // s: j
    // q: i
    // p: j
    // r: i

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

int main() { /* placeholder */}
