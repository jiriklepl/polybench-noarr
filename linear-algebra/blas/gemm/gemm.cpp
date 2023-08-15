#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "gemm.hpp"

// autotuning
#include "noarr/structures/extra/shortcuts.hpp"
#include "test.hpp"

using num_t = DATA_TYPE;

namespace {

struct tuning {
	NOARR_TUNE_BEGIN(opentuner_formatter( \
		std::cout, \
		std::make_shared<noarr::tuning::cmake_compile_command_builder>("../..", "build", "gemm", "-DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -D_POSIX_C_SOURCE=200809L"), \
		std::make_shared<noarr::tuning::direct_run_command_builder>("build/gemm"), \
		"return Result(time=float(run_result['stderr'].split()[0]))"));

	NOARR_TUNE_PAR(block_i, noarr::tuning::choice,
		noarr::bcast<'I'>(noarr::lit<1>),
		noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<2>),
		noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<4>),
		noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<8>),
		noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<16>),
		noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<32>),
		noarr::strip_mine<'i', 'I', 'i'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_j, noarr::tuning::choice,
		noarr::bcast<'J'>(noarr::lit<1>),
		noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<2>),
		noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<4>),
		noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<8>),
		noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<16>),
		noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<32>),
		noarr::strip_mine<'j', 'J', 'j'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_k, noarr::tuning::choice,
		noarr::bcast<'K'>(noarr::lit<1>),
		noarr::strip_mine<'k', 'K', 'k'>(noarr::lit<2>),
		noarr::strip_mine<'k', 'K', 'k'>(noarr::lit<4>),
		noarr::strip_mine<'k', 'K', 'k'>(noarr::lit<8>),
		noarr::strip_mine<'k', 'K', 'k'>(noarr::lit<16>),
		noarr::strip_mine<'k', 'K', 'k'>(noarr::lit<32>),
		noarr::strip_mine<'k', 'K', 'k'>(noarr::lit<64>));

	NOARR_TUNE_PAR(order, noarr::tuning::choice,
		*block_i ^ *block_j ^ *block_k,
		*block_i ^ *block_k ^ *block_j,
		*block_j ^ *block_i ^ *block_k,
		*block_j ^ *block_k ^ *block_i,
		*block_k ^ *block_i ^ *block_j,
		*block_k ^ *block_j ^ *block_i);

	NOARR_TUNE_END();
} tuning;

// initialization function
[[gnu::cold]]
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) noexcept {
	// C: i x j
	// A: i x k
	// B: k x j

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	noarr::traverser(C)
		.for_each([=](auto state) noexcept {
			auto [i, j] = noarr::get_indices<'i', 'j'>(state);
			C[state] = (num_t)((i * j + 1) % (C | noarr::get_length<'i'>())) / (C | noarr::get_length<'i'>());
		});

	noarr::traverser(A)
		.for_each([=](auto state) noexcept {
			auto [i, k] = noarr::get_indices<'i', 'k'>(state);
			A[state] = (num_t)(i * (k + 1) % (A | noarr::get_length<'k'>())) / (A | noarr::get_length<'k'>());
		});
	
	noarr::traverser(B)
		.for_each([=](auto state) noexcept {
			auto [k, j] = noarr::get_indices<'k', 'j'>(state);
			B[state] = (num_t)(k * (j + 2) % (B | noarr::get_length<'j'>())) / (B | noarr::get_length<'j'>());
		});
}

// computation kernel
[[gnu::hot]]
void kernel_gemm(num_t alpha, num_t beta, auto C, auto A, auto B) noexcept {
	// C: i x j
	// A: i x k
	// B: k x j

	noarr::traverser(C)
		.for_each([=](auto state) noexcept {
			C[state] *= beta;
		});

	noarr::traverser(C, A, B).order(*tuning.order)
		.for_each([=](auto state) noexcept {
			C[state] += alpha * A[state] * B[state];
		});
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nj = NJ;
	std::size_t nk = NK;

	// input data
	num_t alpha;
	num_t beta;

	auto C = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(ni, nj));
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'k'>(ni, nk));
	auto B = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'k', 'j'>(nk, nj));

	// initialize data
	init_array(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_gemm(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, C.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << duration.count() << std::endl;
}
