#include <chrono>
#include <iostream>

#include "noarr/structures/extra/shortcuts.hpp"
#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"
#include "noarr/structures/interop/bag.hpp"
#include "noarr/structures/interop/serialize_data.hpp"

using num_t = float;

namespace {

// initialization function
void init_array(auto A) {
    // A: i x j

    auto n = noarr::get_length<'i'>(A);

    noarr::traverser(A)
        .template for_dims<'i'>([=](auto inner) {
            auto state = inner.state();

            auto i = noarr::get_index<'i'>(state);

            inner
                .order(noarr::slice<'j'>(0, i))
                .template for_each<'j'>([=](auto state) {
                    A[state] = (num_t) (-noarr::get_index<'j'>(state) % n) / n + 1;
                });
            
            inner
                .order(noarr::shift<'j'>(0, i + 1))
                .template for_each<'j'>([=](auto state) {
                    A[state] = 0;
                });
            
            A[state & noarr::idx<'j'>(i)] = 1;
        });
}

// computation kernel
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

int main(int argc, char *argv[]) {
    using namespace std::string_literals;

    // problem size
    std::size_t n = N;

    // data
    auto A = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(n, n));

    // initialize data
    init_array(A.get_ref());

    auto start = std::chrono::high_resolution_clock::now();

    // run kernel
    kernel_lu(A.get_ref());

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // print results
    if (argv[0] != ""s)
        noarr::serialize_data(std::cout, A);

    std::cout << duration.count() << std::endl;
}
