#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/planner.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "doitgen.hpp"

// autotuning
#include <noarr/structures/tuning/formatters/opentuner_formatter.hpp>

using num_t = DATA_TYPE;

namespace {

constexpr auto r_vec =  noarr::vector<'r'>();
constexpr auto q_vec =  noarr::vector<'q'>();
constexpr auto p_vec =  noarr::vector<'p'>();
constexpr auto s_vec =  noarr::vector<'s'>();

struct tuning {
	NOARR_TUNE_BEGIN(noarr::tuning::opentuner_formatter(
		std::cout,
		std::make_shared<noarr::tuning::cmake_compile_command_builder>("../..", "build", "doitgen", "-DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE"),
		std::make_shared<noarr::tuning::direct_run_command_builder>("build/doitgen"),
		"return Result(time=float(run_result['stderr'].split()[0]))"));

	NOARR_TUNE_PAR(block_r, noarr::tuning::choice,
		noarr::bcast<'R'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'r', 'R', 'r', 'x'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'r', 'R', 'r', 'x'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'r', 'R', 'r', 'x'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'r', 'R', 'r', 'x'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'r', 'R', 'r', 'x'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'r', 'R', 'r', 'x'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_q, noarr::tuning::choice,
		noarr::bcast<'Q'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'q', 'Q', 'q', 'y'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'q', 'Q', 'q', 'y'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'q', 'Q', 'q', 'y'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'q', 'Q', 'q', 'y'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'q', 'Q', 'q', 'y'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'q', 'Q', 'q', 'y'>(noarr::lit<64>));

	NOARR_TUNE_PAR(order, noarr::tuning::choice,
		*block_r ^ *block_q,
		*block_q ^ *block_r);

	NOARR_TUNE_PAR(a_layout, noarr::tuning::choice,
		r_vec ^ q_vec ^ p_vec,
		r_vec ^ p_vec ^ q_vec,
		q_vec ^ r_vec ^ p_vec,
		q_vec ^ p_vec ^ r_vec,
		p_vec ^ r_vec ^ q_vec,
		p_vec ^ q_vec ^ r_vec);

	NOARR_TUNE_PAR(c4_layout, noarr::tuning::choice,
		s_vec ^ p_vec,
		p_vec ^ s_vec);

	NOARR_TUNE_END();
} tuning;

// initialization function
void init_array(auto A, auto C4) {
	// A: r x q x p
	// C4: s x p

	auto np = A | noarr::get_length<'p'>();

	noarr::traverser(A)
		.for_each([=](auto state) {
			auto [r, q, p] = noarr::get_indices<'r', 'q', 'p'>(state);
			A[state] = (num_t)((r * q + p) % np) / np;
		});

	noarr::traverser(C4)
		.for_each([=](auto state) {
			auto [s, p] = noarr::get_indices<'s', 'p'>(state);
			C4[state] = (num_t)(s * p % np) / np;
		});
}

// computation kernel
template<class Order = noarr::neutral_proto>
void kernel_doitgen(auto A, auto C4, auto sum, Order order = {}) {
	// A: r x q x p
	// C4: s x p
	// sum: p

	auto A_rqs = A ^ noarr::rename<'p', 's'>();

	noarr::planner(A, C4, sum)
		.template for_sections<'r', 'q'>([=](auto inner) {
			inner.template for_sections<'p'>([=](auto inner) {
				auto state = inner.state();

				sum[state] = 0;

				inner.for_each([=](auto state) {
					sum[state] += A_rqs[state] * C4[state];
				})
				();
			})
			();

			inner
				.order(noarr::reorder<'p'>())
				.for_each([=](auto state) {
					A[state] = sum[state];
				})
				();
		})
		.order(noarr::hoist<'p'>())
		.order(noarr::hoist<'q'>())
		.order(noarr::hoist<'r'>())
		.order(order)
		();
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t nr = NR;
	std::size_t nq = NQ;
	std::size_t np = NP;

	auto set_sizes = noarr::set_length<'r'>(nr) ^ noarr::set_length<'q'>(nq) ^ noarr::set_length<'s'>(np) ^ noarr::set_length<'p'>(np);

	// data
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.a_layout ^ set_sizes);
	auto sum = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'p'>(np));
	auto C4 = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.c4_layout ^ set_sizes);

	// initialize data
	init_array(A.get_ref(), C4.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_doitgen(A.get_ref(), C4.get_ref(), sum.get_ref(), *tuning.order);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, A.get_ref() ^ noarr::reorder<'r', 'q', 'p'>());
	}

	std::cerr << duration.count() << std::endl;
}
