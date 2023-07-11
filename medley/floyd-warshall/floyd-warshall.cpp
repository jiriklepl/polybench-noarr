#include <algorithm>

#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

// initialization function
void init_array(auto path) {
    // path: i x j

    auto n = path | noarr::get_length<'i'>();

    noarr::traverser(path)
        .for_each([=](auto state) {
            auto [i, j] = state | noarr::get_indices<'i', 'j'>(state);

            path[state] = i * j % 7 + 1;

            if ((i + j) % 13 == 0 || (i + j) % 7 == 0 || (i + j) % 11 == 0)
                path[state] = 999;
        });
}


// computation kernel
void kernel_floyd_warshall(auto path) {
    // path: i x j

    auto path_start_k = path ^ noarr::rename<'i', 'k'>();
    auto path_end_k = path ^ noarr::rename<'j', 'k'>();
    
    noarr::traverser(path, path_start_k, path_end_k)
        .template for_dims<'k'>([=](auto inner) {
            inner.for_each([=](auto state) {
                path[state] = std::min(path_start_k[state] + path_end_k[state], path[state]);
            });

        });
}

} // namespace

int main() { /* placeholder */}
