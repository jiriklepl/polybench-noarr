#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "floyd-warshall.hpp"
#include "noarr/structures/structs/blocks.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec =  noarr::vector<'i'>();
constexpr auto j_vec =  noarr::vector<'j'>();
constexpr auto k_vec =  noarr::vector<'k'>();

struct tuning {
	DEFINE_PROTO_STRUCT(block_i, noarr::hoist<'i'>());
	DEFINE_PROTO_STRUCT(block_j, noarr::hoist<'j'>());
	DEFINE_PROTO_STRUCT(block_k, noarr::hoist<'k'>());

	DEFINE_PROTO_STRUCT(order, block_j ^ block_i ^ block_k);

	DEFINE_PROTO_STRUCT(path_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto path) {
	// path: i x j

	noarr::traverser(path)
		.for_each([=](auto state) {
			auto [i, j] = noarr::get_indices<'i', 'j'>(state);

			path[state] = i * j % 7 + 1;

			if ((i + j) % 13 == 0 || (i + j) % 7 == 0 || (i + j) % 11 == 0)
				path[state] = 999;
		});
}


// computation kernel
template<class Order = noarr::neutral_proto>
[[gnu::flatten, gnu::noinline]]
void kernel_floyd_warshall(auto path, Order order = {}) {
	// path: i x j

	auto path_start_k = path ^ noarr::rename<'i', 'k'>();
	auto path_end_k = path ^ noarr::rename<'j', 'k'>();

	#pragma scop
	noarr::traverser(path, path_start_k, path_end_k)
		.order(noarr::hoist<'k'>())
		.order(order)
		.for_each([=](auto state) {
			path[state] = std::min(path_start_k[state] + path_end_k[state], path[state]);
		});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// data
	auto path = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.path_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));

	// initialize data
	init_array(path.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_floyd_warshall(path.get_ref(), tuning.order);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, path.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
