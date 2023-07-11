#include <chrono>
#include <iostream>

#include <noarr/structures/extra/shortcuts.hpp>
#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "doitgen.hpp"

using num_t = DATA_TYPE;

namespace {

// initialization function
void init_array(auto A, auto C4) {
    // A: r x q x p
    // C4: s x p

    auto np = A | noarr::get_length<'p'>();

    noarr::traverser(A)
        .for_each([=](auto state) {
            auto [r, q, p] = noarr::get_indices<'r', 'q', 'p'>(state);
            A[state] = (num_t)((r * q + p) % np) / np;
        });

    noarr::traverser(C4)
        .for_each([=](auto state) {
            auto [s, p] = noarr::get_indices<'s', 'p'>(state);
            C4[state] = (num_t)(s * p % np) / np;
        });
}

// computation kernel
void kernel_doitgen(auto A, auto C4, auto sum) {
    // A: r x q x p
    // C4: s x p
    // sum: p

    auto A_rqs = A ^ noarr::rename<'p', 's'>();

    noarr::traverser(A, C4, sum)
        .template for_dims<'r', 'q'>([=](auto inner) {
            inner.template for_dims<'p'>([=](auto inner) {
                auto state = inner.state();

                sum[state] = 0;

                inner.template for_each<'s'>([=](auto state) {
                    sum[state] += A_rqs[state] * C4[state];
                });
            });

            inner.template for_each<'p'>([=](auto state) {
                A[state] = sum[state];
            });
        });
}

} // namespace

int main(int argc, char *argv[]) {
    using namespace std::string_literals;

    // problem size
    std::size_t nr = NR;
    std::size_t nq = NQ;
    std::size_t np = NP;

    // data
    auto A = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'r', 'q', 'p'>(nr, nq, np));
    auto sum = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'p'>(np));
    auto C4 = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'s', 'p'>(np, np));

    // initialize data
    init_array(A.get_ref(), C4.get_ref());

    auto start = std::chrono::high_resolution_clock::now();

    // run kernel
    kernel_doitgen(A.get_ref(), C4.get_ref(), sum.get_ref());

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // print results
    if (argv[0] != ""s)
        noarr::serialize_data(std::cout, A);

    std::cout << duration.count() << std::endl;
}
