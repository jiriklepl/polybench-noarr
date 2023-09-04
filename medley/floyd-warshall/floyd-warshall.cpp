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

// autotuning
#include "test.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec =  noarr::vector<'i'>();
constexpr auto j_vec =  noarr::vector<'j'>();
constexpr auto k_vec =  noarr::vector<'k'>();

struct tuning {
	NOARR_TUNE_BEGIN(opentuner_formatter( \
		std::cout, \
		std::make_shared<noarr::tuning::cmake_compile_command_builder>("../..", "build", "floyd-warshall", "-DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE"), \
		std::make_shared<noarr::tuning::direct_run_command_builder>("build/floyd-warshall"), \
		"return Result(time=float(run_result['stderr'].split()[0]))"));

	NOARR_TUNE_PAR(block_i, noarr::tuning::choice,
		noarr::bcast<'I'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_j, noarr::tuning::choice,
		noarr::bcast<'J'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_k, noarr::tuning::choice,
		noarr::bcast<'K'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<64>));

	NOARR_TUNE_PAR(order, noarr::tuning::choice,
		*block_k ^ *block_i ^ *block_j,
		*block_k ^ *block_j ^ *block_i,
		*block_i ^ *block_k ^ *block_j,
		*block_i ^ *block_j ^ *block_k,
		*block_j ^ *block_k ^ *block_i,
		*block_j ^ *block_i ^ *block_k);

	NOARR_TUNE_PAR(path_layout, noarr::tuning::choice,
		i_vec ^ j_vec,
		j_vec ^ i_vec);

	NOARR_TUNE_END();
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
void kernel_floyd_warshall(auto path, Order order = {}) {
	// path: i x j

	auto path_start_k = path ^ noarr::rename<'i', 'k'>();
	auto path_end_k = path ^ noarr::rename<'j', 'k'>();
	
	noarr::traverser(path, path_start_k, path_end_k)
		.order(noarr::hoist<'k'>())
		.order(order)
		.for_each([=](auto state) {
			path[state] = std::min(path_start_k[state] + path_end_k[state], path[state]);
		});
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// data
	auto path = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.path_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));

	// initialize data
	init_array(path.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_floyd_warshall(path.get_ref(), *tuning.order);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, path.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << duration.count() << std::endl;
}
