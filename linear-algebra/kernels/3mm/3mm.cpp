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
void init_array(auto A, auto B, auto C, auto D) {
    // A: i x k
    // B: k x j
    // C: j x m
    // D: m x l

    auto ni = A | noarr::get_length<'i'>();
    auto nj = B | noarr::get_length<'j'>();
    auto nk = A | noarr::get_length<'k'>();
    auto nm = C | noarr::get_length<'m'>();
    auto nl = D | noarr::get_length<'l'>();

    noarr::traverser(A)
        .for_each([=](auto state) {
            auto [i, k] = noarr::get_indices<'i', 'k'>(state);
            A[state] = (num_t)((i * k + 1) % ni) / (5 * ni);
        });

    noarr::traverser(B)
        .for_each([=](auto state) {
            auto [k, j] = noarr::get_indices<'k', 'j'>(state);
            B[state] = (num_t)((k * (j + 1) + 2) % nj) / (5 * nj);
        });

    noarr::traverser(C)
        .for_each([=](auto state) {
            auto [j, m] = noarr::get_indices<'j', 'm'>(state);
            C[state] = (num_t)(j * (m + 3) % nl) / (5 * nl);
        });

    noarr::traverser(D)
        .for_each([=](auto state) {
            auto [m, l] = noarr::get_indices<'m', 'l'>(state);
            D[state] = (num_t)((m * (l + 2) + 2) % nk) / (5 * nk);
        });
}

// computation kernel
void kernel_3mm(auto E, auto A, auto B, auto F, auto C, auto D, auto G) {
    // E: i x j
    // A: i x k
    // B: k x j
    // F: j x l
    // C: j x m
    // D: m x l
    // G: i x l

    noarr::traverser(E, A, B)
        .template for_dims<'i', 'j'>([=](auto inner) {
            auto state = inner.state();

            E[state] = 0;

            inner.template for_each<'k'>([=](auto state) {
                E[state] += A[state] * B[state];
            });
        });

    noarr::traverser(F, C, D)
        .template for_dims<'j', 'l'>([=](auto inner) {
            auto state = inner.state();

            F[state] = 0;

            inner.template for_each<'m'>([=](auto state) {
                F[state] += C[state] * D[state];
            });
        });

    noarr::traverser(G, E, F)
        .template for_dims<'i', 'l'>([=](auto inner) {
            auto state = inner.state();

            G[state] = 0;

            inner.template for_each<'j'>([=](auto state) {
                G[state] += E[state] * F[state];
            });
        });
}

} // namespace

int main(int argc, char *argv[]) {
    using namespace std::string_literals;

    // problem size
    std::size_t ni = NI;
    std::size_t nj = NJ;
    std::size_t nk = NK;
    std::size_t nl = NL;
    std::size_t nm = NM;

    // data
    auto E = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(ni, nj));
    auto A = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'k'>(ni, nk));
    auto B = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'k', 'j'>(nk, nj));

    auto F = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'j', 'l'>(nj, nl));
    auto C = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'j', 'm'>(nj, nm));
    auto D = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'m', 'l'>(nm, nl));

    auto G = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'l'>(ni, nl));

    // initialize data
    init_array(A.get_ref(), B.get_ref(), C.get_ref(), D.get_ref());

    auto start = std::chrono::high_resolution_clock::now();

    // run kernel
    kernel_3mm(E.get_ref(), A.get_ref(), B.get_ref(), F.get_ref(), C.get_ref(), D.get_ref(), G.get_ref());

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // print results
    if (argv[0] != ""s)
        noarr::serialize_data(std::cout, /*...*/);

    std::cout << duration.count() << std::endl;
}
