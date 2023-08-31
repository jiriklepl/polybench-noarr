#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/planner.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "atax.hpp"
#include "noarr/structures/structs/blocks.hpp"

// autotuning
#include "test.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec =  noarr::vector<'i'>();
constexpr auto j_vec =  noarr::vector<'j'>();

struct tuning {
	NOARR_TUNE_BEGIN(opentuner_formatter( \
		std::cout, \
		std::make_shared<noarr::tuning::cmake_compile_command_builder>("../..", "build", "gemm", "-DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -D_POSIX_C_SOURCE=200809L"), \
		std::make_shared<noarr::tuning::direct_run_command_builder>("build/gemm"), \
		"return Result(time=float(run_result['stderr'].split()[0]))"));

	NOARR_TUNE_PAR(block_i1, noarr::tuning::choice,
		noarr::bcast<'I'>(noarr::lit<1>),
		noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<2>),
		noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<4>),
		noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<8>),
		noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<16>),
		noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<32>),
		noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_j1, noarr::tuning::choice,
		noarr::bcast<'J'>(noarr::lit<1>),
		noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<2>),
		noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<4>),
		noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<8>),
		noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<16>),
		noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<32>),
		noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<64>));

	NOARR_TUNE_PAR(order1, noarr::tuning::choice,
		*block_i1 ^ *block_j1,
		*block_j1 ^ *block_i1);

	NOARR_TUNE_PAR(block_i2, noarr::tuning::choice,
		noarr::bcast<'I'>(noarr::lit<1>),
		noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<2>),
		noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<4>),
		noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<8>),
		noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<16>),
		noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<32>),
		noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_j2, noarr::tuning::choice,
		noarr::bcast<'J'>(noarr::lit<1>),
		noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<2>),
		noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<4>),
		noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<8>),
		noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<16>),
		noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<32>),
		noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<64>));

	NOARR_TUNE_PAR(order2, noarr::tuning::choice,
		*block_i2 ^ *block_j2,
		*block_j2 ^ *block_i2);

	NOARR_TUNE_PAR(c_layout, noarr::tuning::choice,
		i_vec ^ j_vec,
		j_vec ^ i_vec);

	NOARR_TUNE_END();
} tuning;

// initialization function
void init_array(auto A, auto x) {
	// A: i x j
	// x: j

	auto ni = A | noarr::get_length<'i'>();
	auto nj = A | noarr::get_length<'j'>();

	noarr::traverser(x).for_each([=](auto state) {
		auto j = noarr::get_index<'j'>(state);
		x[state] = 1 + j / (num_t)nj;
	});

	noarr::traverser(A).for_each([=](auto state) {
		auto [i, j] = noarr::get_indices<'i', 'j'>(state);
		A[state] = (num_t)((i + j) % nj) / (5 * ni);
	});
}

// computation kernel
template<class Order1 = noarr::neutral_proto, class Order2 = noarr::neutral_proto>
void kernel_atax(auto A, auto x, auto y, auto tmp, Order1 order1 = {}, Order2 order2 = {}) {
	// A: i x j
	// x: j
	// y: j
	// tmp: i

	noarr::traverser(y).for_each([=](auto state) {
		y[state] = 0;
	});

	noarr::traverser(tmp).for_each([=](auto state) {
		tmp[state] = 0;
	});

	noarr::traverser(tmp, A, x)
		.order(order1)
		.for_each([=](auto state) {
			tmp[state] += A[state] * x[state];
		});

	noarr::traverser(y, A, tmp)
		.order(order2)
		.for_each([=](auto state) {
			y[state] += A[state] * tmp[state];
		});
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nj = NJ;

	// data
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.c_layout ^ noarr::set_length<'i'>(ni) ^ noarr::set_length<'j'>(nj));

	auto x = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'j'>(nj));
	auto y = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'j'>(nj));

	auto tmp = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(ni));

	// initialize data
	init_array(A.get_ref(), x.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_atax(A.get_ref(), x.get_ref(), y.get_ref(), tmp.get_ref(), *tuning.order1, *tuning.order2);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, y);
	}

	std::cerr << duration.count() << std::endl;
}
