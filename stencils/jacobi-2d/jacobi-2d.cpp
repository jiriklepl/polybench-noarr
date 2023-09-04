#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "jacobi-2d.hpp"

// autotuning
#include <noarr/structures/tuning/formatters/opentuner_formatter.hpp>

using num_t = DATA_TYPE;

namespace {

struct tuning {
	NOARR_TUNE_BEGIN(noarr::tuning::opentuner_formatter(
		std::cout,
		std::make_shared<noarr::tuning::cmake_compile_command_builder>("../..", "build", "jacobi-2d", "-DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE"),
		std::make_shared<noarr::tuning::direct_run_command_builder>("build/jacobi-2d"),
		"return Result(time=float(run_result['stderr'].split()[0]))"));

	NOARR_TUNE_PAR(block_i, noarr::tuning::choice,
		noarr::bcast<'I'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 'a'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 'a'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 'a'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 'a'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 'a'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 'a'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_j, noarr::tuning::choice,
		noarr::bcast<'J'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 'b'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 'b'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 'b'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 'b'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 'b'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 'b'>(noarr::lit<64>));

	NOARR_TUNE_PAR(order, noarr::tuning::choice,
		*block_i ^ *block_j,
		*block_i ^ *block_j);

	NOARR_TUNE_END();
} tuning;


// initialization function
[[gnu::cold]]
void init_array(auto A, auto B) {
	// A: i x j
	// B: i x j

	auto n = A | noarr::get_length<'i'>();

	noarr::traverser(A, B)
		.for_each([=](auto state) {
			auto [i, j] = noarr::get_indices<'i', 'j'>(state);

			A[state] = ((num_t)i * (j + 2) + 2) / n;
			B[state] = ((num_t)i * (j + 3) + 3) / n;
		});
}


// computation kernel
template<class Order = noarr::neutral_proto>
[[gnu::hot]]
void kernel_jacobi_2d(std::size_t steps, auto A, auto B, Order order = {}) noexcept {
	// A: i x j
	// B: i x j

	auto traverser = noarr::traverser(A, B).order(noarr::bcast<'t'>(steps));

	traverser
		.order(noarr::symmetric_spans<'i', 'j'>(traverser.top_struct(), 1, 1))
		.order(order)
		.template for_dims<'t'>([=](auto inner) noexcept {
			inner.for_each([=](auto state) {
				B[state] = (num_t).2 * (
					A[state] +
					A[neighbor<'j'>(state, -1)] +
					A[neighbor<'j'>(state, +1)] +
					A[neighbor<'i'>(state, +1)] +
					A[neighbor<'i'>(state, -1)]);
			});

			inner.for_each([=](auto state) noexcept {
				A[state] = (num_t).2 * (
					B[state] +
					B[neighbor<'j'>(state, -1)] +
					B[neighbor<'j'>(state, +1)] +
					B[neighbor<'i'>(state, +1)] +
					B[neighbor<'i'>(state, -1)]);
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
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(n, n));
	auto B = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(n, n));

	// initialize data
	init_array(A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_jacobi_2d(t, A.get_ref(), B.get_ref(), *tuning.order);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, A.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << duration.count() << std::endl;
}
