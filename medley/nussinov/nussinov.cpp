#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures/extra/shortcuts.hpp>
#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "nussinov.hpp"

using num_t = DATA_TYPE;
using base_t = char;

namespace {

// initialization function
void init_array(auto seq, auto table) {
    // seq: i
    // table: i x j

    noarr::traverser(seq)
        .for_each([=](auto state) {
            auto i = noarr::get_index<'i'>(state);

            seq[state] = (base_t)((i + 1) % 4);
        });
    
    noarr::traverser(table)
        .for_each([=](auto state) {
            table[state] = 0;
        });
}

// computation kernel
void kernel_nussinov(auto seq, auto table) {
    // seq: i
    // table: i x j

    auto seq_j = seq ^ noarr::rename<'i', 'j'>();
    auto table_ik = table ^ noarr::rename<'j', 'k'>();
    auto table_kj = table ^ noarr::rename<'i', 'k'>();

    noarr::traverser(seq, table, table_ik, table_kj)
        .order(noarr::reverse<'i'>())
        .template for_dims<'i'>([=](auto inner) {
            auto state = inner.state();

            inner
                .order(noarr::shift<'j'>(noarr::get_index<'i'>(state) + 1))
                .template for_dims<'j'>([=](auto inner) {
                    auto state = inner.state();

                    if (noarr::get_index<'j'>(state) >= 0)
                        table[state] = std::max(
                            table[state],
                            table[noarr::neighbor<'j'>(state, -1)]);

                    if (noarr::get_index<'i'>(state) + 1 < (table | noarr::get_length<'i'>()))
                        table[state] = std::max(
                            table[state],
                            table[noarr::neighbor<'i'>(state, 1)]);

                    if (noarr::get_index<'j'>(state) >= 0
                     || noarr::get_index<'i'>(state) + 1 < (table | noarr::get_length<'i'>())) {
                        if (noarr::get_index<'i'>(state) < noarr::get_index<'j'>(state) - 1)
                            table[state] = std::max(
                                table[state],
                                table[noarr::neighbor<'i', 'j'>(state, 1, -1)]
                                + (seq[state] + seq_j[state] == 3 ? 1 : 0));
                        else
                            table[state] = std::max(
                                table[state],
                                table[noarr::neighbor<'i', 'j'>(state, 1, -1)]);
                    }

                    inner
                        .order(noarr::span<'k'>(noarr::get_index<'i'>(state) + 1, noarr::get_index<'j'>(state)))
                        .template for_each<'k'>([=](auto state) {
                            table[state] = std::max(
                                table[state],
                                table_ik[state] + table_kj[noarr::neighbor<'k'>(state, 1)]);
                        });
                });
        });
}

} // namespace

int main(int argc, char *argv[]) {
    using namespace std::string_literals;

    // problem size
    std::size_t n = N;

    // data
    auto seq = noarr::make_bag(noarr::scalar<base_t>() ^ noarr::sized_vector<'i'>(n));
    auto table = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(n, n));

    // initialize data
    init_array(seq.get_ref(), table.get_ref());

    auto start = std::chrono::high_resolution_clock::now();

    // run kernel
    kernel_nussinov(seq.get_ref(), table.get_ref());

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // print results
    if (argv[0] != ""s) [table = table.get_ref()] {
        std::cout << std::fixed << std::setprecision(2);
        noarr::traverser(table)
            .template for_dims<'i'>([=](auto inner) {
                auto state = inner.state();
                std::cout << std::fixed << std::setprecision(2);
                inner
                    .order(noarr::shift<'j'>(noarr::get_index<'i'>(state)))
                    .template for_each<'j'>([=](auto state) {
                        std::cout << table[state] << " ";
                    });
            });
    }();


    std::cerr << duration << std::endl;
}
