#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

// initialization function
void init_array(auto x1, auto x2, auto y1, auto y2, auto A) {
    // x1: i
    // x2: i
    // y1: j
    // y2: j
    // A: i x j

    auto n = A | noarr::get_length<'i'>();

    noarr::traverser(x1, x2, y1, y2, A)
        .template for_dims<'i'>([=](auto inner) {
            auto state = inner.state();
            auto i = noarr::get_index<'i'>(state);
            
            x1[state] = (num_t)(i % n) / n;
            x2[state] = (num_t)((i + 1) % n) / n;
            y1[state] = (num_t)((i + 3) % n) / n;
            y2[state] = (num_t)((i + 4) % n) / n;

            inner.template for_each<'j'>([=](auto state) {
                auto j = noarr::get_index<'j'>(state);

                A[state] = (num_t)(i * j % n) / n;
            });
        });
}

// computation kernel
void kernel_mvt(auto x1, auto x2, auto y1, auto y2, auto A) {
    // x1: i
    // x2: i
    // y1: j
    // y2: j
    // A: i x j
    auto A_ji = A | noarr::rename<'i', 'j', 'j', 'i'>();

    noarr::traverser(x1, A, y1)
        .template for_each<'i', 'j'>([=](auto state) {
            x1[state] += A[state] * y1[state];
        });

    noarr::traverser(x2, A, y2)
        .template for_each<'i', 'j'>([=](auto state) {
            x2[state] += A_ji[state] * y2[state];
        });
}

} // namespace

int main() { /* placeholder */}
