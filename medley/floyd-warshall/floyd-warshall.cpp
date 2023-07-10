#include <algorithm>

#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_floyd_warshall(auto path) {
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
