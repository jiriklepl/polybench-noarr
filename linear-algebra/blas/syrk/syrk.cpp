#include <chrono>
#include <iostream>

#include <noarr/structures/extra/shortcuts.hpp>
#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "syrk.hpp"

using num_t = DATA_TYPE;

namespace {

// initialization function
void init_array(num_t &alpha, num_t &beta, auto C, auto A) {
    alpha = 1.5;
    beta = 1.2;

    auto ni = C | noarr::get_length<'i'>();
    auto nk = A | noarr::get_length<'k'>();

    noarr::traverser(A)
        .for_each([=](auto state) {
            auto [i, k] = noarr::get_indices<'i', 'k'>(state);

            A[state] = (num_t)((i * k + 1) % ni) / ni;
        });

    noarr::traverser(C)
        .for_each([=](auto state) {
            auto [i, j] = noarr::get_indices<'i', 'j'>(state);

            C[state] = (num_t)((i * j + 2) % nk) / nk;
        });
}

// computation kernel
void kernel_syrk(num_t alpha, num_t beta, auto C, auto A) {
    // C: i x j
    // A: i x k
    auto A_renamed = A ^ noarr::rename<'i', 'j'>();

    noarr::traverser(C, A)
        .template for_dims<'i'>(
            [=](auto inner) {
                auto state = inner.state();

                inner
                    .order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state)))
                    .template for_each<'j'>([=](auto state) {
                        C[state] *= beta;
                    });

                inner
                    .order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state)))
                    .template for_each<'k', 'j'>([=](auto state) {
                        C[state] += alpha * A_renamed[state] * A[state];
                    });
            });
}

} // namespace

int main(int argc, char *argv[]) {
    using namespace std::string_literals;

    // problem size
    std::size_t ni = NI;
    std::size_t nk = NK;

    // data
    num_t alpha;
    num_t beta;

    auto C = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(ni, ni));
    auto A = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'k'>(ni, nk));

    // initialize data
    init_array(alpha, beta, C.get_ref(), A.get_ref());

    auto start = std::chrono::high_resolution_clock::now();

    // run kernel
    kernel_syrk(alpha, beta, C.get_ref(), A.get_ref());

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // print results
    if (argv[0] != ""s)
        noarr::serialize_data(std::cout, C.get_ref());

    std::cerr << duration << std::endl;
}
