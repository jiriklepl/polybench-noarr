#include <chrono>
#include <iostream>

#include <noarr/structures/extra/shortcuts.hpp>
#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "adi.hpp"

using num_t = DATA_TYPE;

namespace {

// initialization function
void init_array(auto u) {
    // u: i x j

    auto n = u | noarr::get_length<'i'>();

    noarr::traverser(u)
        .for_each([=](auto state) {
            auto [i, j] = noarr::get_indices<'i', 'j'>(state);

            u[state] = (num_t)(i + n - j) / n;
        });
}

// computation kernel
void kernel_adi(auto steps, auto u, auto v, auto p, auto q) {
    // u: i x j
    // v: j x i
    // p: i x j
    // q: i x j

    auto u_trans = u ^ noarr::rename<'i', 'j', 'j', 'i'>();
    auto v_trans = v ^ noarr::rename<'i', 'j', 'j', 'i'>();
    auto traverser = noarr::traverser(u, v, p, q).order(noarr::bcast<'t'>(steps));

    num_t DX = (num_t)1.0 / (traverser.top_struct() | noarr::get_length<'i'>());
    num_t DY = (num_t)1.0 / (traverser.top_struct() | noarr::get_length<'j'>());
    num_t DT = (num_t)1.0 / (traverser.top_struct() | noarr::get_length<'t'>());

    num_t B1 = 2.0;
    num_t B2 = 1.0;

    num_t mul1 = B1 * DT / (DX * DX);
    num_t mul2 = B2 * DT / (DY * DY);

    num_t a = -mul1 / (num_t)2.0;
    num_t b = (num_t)1.0 + mul1;
    num_t c = a;

    num_t d = -mul2 / (num_t)2.0;
    num_t e = (num_t)1.0 + mul2;
    num_t f = d;

    traverser.order(noarr::symmetric_spans<'i', 'j'>(traverser.top_struct(), 1, 1))
        .template for_dims<'t'>([=](auto inner) {
            // column sweep
            inner.template for_dims<'i'>([=](auto inner) {
                auto state = inner.state();

                v[state & noarr::idx<'j'>(0)] = (num_t)1.0;
                p[state & noarr::idx<'j'>(0)] = (num_t)0.0;
                q[state & noarr::idx<'j'>(0)] = v[state & noarr::idx<'j'>(0)];

                inner.template for_each<'j'>([=](auto state) {
                    p[state] = -c / (a * p[noarr::neighbor<'j'>(state, -1)] + b);
                    q[state] = (-d * u_trans[noarr::neighbor<'i'>(state, -1)] + (B2 + B1 * d) * u_trans[state] -
                                 f * u_trans[noarr::neighbor<'i'>(state, +1)] -
                                 a * q[noarr::neighbor<'j'>(state, -1)]) /
                               (a * p[noarr::neighbor<'j'>(state, -1)] + b);
                });

                v[state & noarr::idx<'j'>((traverser.top_struct() | noarr::get_length<'j'>()) - 1)] = (num_t)1.0; // TODO: think about this

                inner
                    .order(noarr::reverse<'j'>())
                    .template for_each<'j'>([=](auto state) {
                        v[state] = p[state] * v[noarr::neighbor<'j'>(state, 1)] + q[state];
                    });
            });

            // row sweep
            inner.template for_dims<'i'>([=](auto inner) {
                auto state = inner.state();

                u[state & noarr::idx<'j'>(0)] = (num_t)1.0;
                p[state & noarr::idx<'j'>(0)] = (num_t)0.0;
                q[state & noarr::idx<'j'>(0)] = u[state & noarr::idx<'j'>(0)];

                inner.template for_each<'j'>([=](auto state) {
                    p[state] = -f / (d * p[noarr::neighbor<'j'>(state, -1)] + e);
                    q[state] = (-a * v_trans[noarr::neighbor<'i'>(state, -1)] + (B2 + B1 * a) * v_trans[state] -
                                 c * v_trans[noarr::neighbor<'i'>(state, +1)] -
                                 d * q[noarr::neighbor<'j'>(state, -1)]) /
                               (d * p[noarr::neighbor<'j'>(state, -1)] + e);
                });

                u[state & noarr::idx<'j'>((traverser.top_struct() | noarr::get_length<'j'>()) - 1)] = (num_t)1.0;

                inner
                    .order(noarr::reverse<'j'>())
                    .template for_each<'j'>([=](auto state) {
                        u[state] = p[state] * u[noarr::neighbor<'j'>(state, 1)] + q[state];
                    });
            });
        });
}

} // namespace

int main(int argc, char *argv[]) {
    using namespace std::string_literals;

    // problem size
    std::size_t n = N;
    std::size_t t = TSTEPS;

    // data
    auto u = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(n, n));
    auto v = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'j', 'i'>(n, n));
    auto p = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(n, n));
    auto q = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(n, n));

    // initialize data
    init_array(u.get_ref());

    auto start = std::chrono::high_resolution_clock::now();

    // run kernel
    kernel_adi(t, u.get_ref(), v.get_ref(), p.get_ref(), q.get_ref());

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // print results
    if (argv[0] != ""s)
        noarr::serialize_data(std::cout, u);

    std::cerr << duration << std::endl;
}
