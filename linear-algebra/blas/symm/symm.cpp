#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/planner.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "symm.hpp"

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
		std::make_shared<noarr::tuning::cmake_compile_command_builder>("../..", "build", "symm", "-D" STRINGIFY(DATASET_SIZE) " -D" STRINGIFY(DATA_TYPE_CHOICE)),
		std::make_shared<noarr::tuning::direct_run_command_builder>("build/symm"),
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

	NOARR_TUNE_PAR(order, noarr::tuning::choice,
		*block_i ^ *block_j,
		*block_j ^ *block_i);

	NOARR_TUNE_PAR(c_layout, noarr::tuning::choice,
		i_vec ^ j_vec);

	NOARR_TUNE_PAR(a_layout, noarr::tuning::choice,
		i_vec ^ k_vec);
	
	NOARR_TUNE_PAR(b_layout, noarr::tuning::choice,
		i_vec ^ j_vec);

	NOARR_TUNE_END();
} tuning;

// initialization function
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) noexcept {
	// C: i x j
	// A: i x k
	// B: i x j

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	auto ni = C | noarr::get_length<'i'>();
	auto nj = C | noarr::get_length<'j'>();

	noarr::traverser(C)
		.for_each([=](auto state) constexpr noexcept {
			auto [i, j] = noarr::get_indices<'i', 'j'>(state);
			C[state] = (num_t)((i + j) % 100) / ni;
			B[state] = (num_t)((nj + i - j) % 100) / ni;
		});

	noarr::traverser(A)
		.template for_dims<'i'>([=](auto inner) constexpr noexcept {
			auto state = inner.state();
			
			inner.order(noarr::slice<'k'>(0, noarr::get_index<'i'>(state) + 1))
				.for_each([=](auto state) constexpr noexcept {
					auto [i, k] = noarr::get_indices<'i', 'k'>(state);
					A[state] = (num_t)((i + k) % 100) / ni;
				});
			
			inner.order(noarr::shift<'k'>(noarr::get_index<'i'>(state) + 1))
				.for_each([=](auto state) constexpr noexcept {
					A[state] = -999;
				});
		});
}

// computation kernel
template<class Order = noarr::neutral_proto>
[[gnu::flatten, gnu::noinline]]
void kernel_symm(num_t alpha, num_t beta, auto C, auto A, auto B, Order order = {}) noexcept {
	// C: i x j
	// A: i x k
	// B: i x j

	auto C_renamed = C ^ noarr::rename<'i', 'k'>();
	auto B_renamed = B ^ noarr::rename<'i', 'k'>();

	noarr::planner(C, A, B)
		.template for_sections<'i', 'j'>([=](auto inner) constexpr noexcept {
			num_t temp = 0;
			auto state = inner.state();

			inner
				.order(noarr::slice<'k'>(0, noarr::get_index<'i'>(state)))
				.for_each([=, &temp](auto state) constexpr noexcept {
					C_renamed[state] += alpha * B[state] * A[state];
					temp += B_renamed[state] * A[state];
				})
				();

			C[state] = beta * C[state] + alpha * B[state] * A[state & noarr::idx<'k'>(noarr::get_index<'i'>(state))] + alpha * temp; // TODO: A_diag
		})
		.order(noarr::hoist<'j'>())
		.order(noarr::hoist<'i'>())
		.order(order)
		();
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nj = NJ;

	// data
	num_t alpha;
	num_t beta;

	auto set_lengths = noarr::set_length<'i'>(ni) ^ noarr::set_length<'k'>(ni) ^ noarr::set_length<'j'>(nj);

	auto C = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.c_layout ^ set_lengths);
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.a_layout ^ set_lengths);
	auto B = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.b_layout ^ set_lengths);

	// initialize data
	init_array(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_symm(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref(), *tuning.order);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, C.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << duration.count() << std::endl;
}
