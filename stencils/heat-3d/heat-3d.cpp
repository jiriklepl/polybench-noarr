#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "heat-3d.hpp"

// autotuning
#include <noarr/structures/tuning/formatters/opentuner_formatter.hpp>

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec =  noarr::vector<'i'>();
constexpr auto j_vec =  noarr::vector<'j'>();
constexpr auto k_vec =  noarr::vector<'k'>();

struct tuning {
	NOARR_TUNE_BEGIN(noarr::tuning::opentuner_formatter(
		std::cout,
		std::make_shared<noarr::tuning::cmake_compile_command_builder>("../..", "build", "heat-3d", "-D" STRINGIFY(DATASET_SIZE) " -D" STRINGIFY(DATA_TYPE_CHOICE)),
		std::make_shared<noarr::tuning::direct_run_command_builder>("build/heat-3d"),
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
		noarr::strip_mine_dynamic<'j', 'J', 'j', 'v'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 'v'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 'v'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 'v'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 'v'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 'v'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_k, noarr::tuning::choice,
		noarr::bcast<'K'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<64>));

	NOARR_TUNE_PAR(order, noarr::tuning::choice,
		*block_i ^ *block_j ^ *block_k,
		*block_i ^ *block_k ^ *block_j,
		*block_j ^ *block_i ^ *block_k,
		*block_j ^ *block_k ^ *block_i,
		*block_k ^ *block_i ^ *block_j,
		*block_k ^ *block_j ^ *block_i);

	NOARR_TUNE_PAR(a_layout, noarr::tuning::choice,
		i_vec ^ j_vec ^ k_vec);

	NOARR_TUNE_PAR(b_layout, noarr::tuning::choice,
		i_vec ^ j_vec ^ k_vec);

	NOARR_TUNE_END();
} tuning;

// initialization function
void init_array(auto A, auto B) noexcept {
	// A: i x j x k
	// B: i x j x k

	auto n = A | noarr::get_length<'i'>();

	noarr::traverser(A, B)
		.for_each([=](auto state) constexpr noexcept {
			auto [i, j, k] = noarr::get_indices<'i', 'j', 'k'>(state);

			A[state] = B[state] = (num_t) (i + j + (n - k)) * 10 / n;
		});
}

// computation kernel
template<class Order = noarr::neutral_proto>
[[gnu::flatten, gnu::noinline]]
void kernel_heat_3d(std::size_t steps, auto A, auto B, Order order = {}) noexcept {
	// A: i x j x k
	// B: i x j x k

	auto traverser = noarr::traverser(A, B).order(noarr::bcast<'t'>(steps));

	traverser
		.order(noarr::symmetric_spans<'i', 'j', 'k'>(traverser.top_struct(), 1, 1, 1))
		.order(order)
		.template for_dims<'t'>([=](auto inner) constexpr noexcept {
			inner.for_each([=](auto state) constexpr noexcept {
				B[state] =
					(num_t).125 * (A[neighbor<'i'>(state, -1)] -
					               2 * A[state] +
								   A[neighbor<'i'>(state, +1)]) +
					(num_t).125 * (A[neighbor<'j'>(state, -1)] -
					               2 * A[state] +
					               A[neighbor<'j'>(state, +1)]) +
					(num_t).125 * (A[neighbor<'k'>(state, -1)] -
					               2 * A[state] +
					               A[neighbor<'k'>(state, +1)]) +
					A[state];
			});

			inner.for_each([=](auto state) constexpr noexcept {
				A[state] =
					(num_t).125 * (B[neighbor<'i'>(state, -1)] -
					               2 * B[state] +
					               B[neighbor<'i'>(state, +1)]) +
					(num_t).125 * (B[neighbor<'j'>(state, -1)] -
					               2 * B[state] +
					               B[neighbor<'j'>(state, +1)]) +
					(num_t).125 * (B[neighbor<'k'>(state, -1)] -
					               2 * B[state] +
					               B[neighbor<'k'>(state, +1)]) +
					B[state];
			});
		});
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;
	std::size_t t = TSTEPS;

	auto set_lengths = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n) ^ noarr::set_length<'k'>(n);

	// data
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.a_layout ^ set_lengths);
	auto B = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.b_layout ^ set_lengths);

	// initialize data
	init_array(A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_heat_3d(t, A.get_ref(), B.get_ref(), *tuning.order);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, A.get_ref() ^ noarr::reorder<'i', 'j', 'k'>());
	}

	std::cerr << duration.count() << std::endl;
}
