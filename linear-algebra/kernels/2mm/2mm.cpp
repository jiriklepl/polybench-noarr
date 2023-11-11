#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/planner.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "2mm.hpp"

// autotuning
#include <noarr/structures/tuning/formatters/opentuner_formatter.hpp>

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec =  noarr::vector<'i'>();
constexpr auto j_vec =  noarr::vector<'j'>();
constexpr auto k_vec =  noarr::vector<'k'>();
constexpr auto l_vec =  noarr::vector<'l'>();

struct tuning {
	NOARR_TUNE_BEGIN(noarr::tuning::opentuner_formatter(
		std::cout,
		std::make_shared<noarr::tuning::cmake_compile_command_builder>("../..", "build", "2mm", "-D" STRINGIFY(DATASET_SIZE) " -D" STRINGIFY(DATA_TYPE_CHOICE)),
		std::make_shared<noarr::tuning::direct_run_command_builder>("build/2mm"),
		"return Result(time=float(run_result['stderr'].split()[0]))"));

	NOARR_TUNE_PAR(block_i1, noarr::tuning::choice,
		noarr::bcast<'I'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_j1, noarr::tuning::choice,
		noarr::bcast<'J'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_i2, noarr::tuning::choice,
		noarr::bcast<'I'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_l2, noarr::tuning::choice,
		noarr::bcast<'L'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'l', 'L', 'l', 't'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'l', 'L', 'l', 't'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'l', 'L', 'l', 't'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'l', 'L', 'l', 't'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'l', 'L', 'l', 't'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'l', 'L', 'l', 't'>(noarr::lit<64>));

	NOARR_TUNE_PAR(order1, noarr::tuning::choice,
		*block_i1 ^ *block_j1,
		*block_j1 ^ *block_i1);

	NOARR_TUNE_PAR(order2, noarr::tuning::choice,
		*block_i2 ^ *block_l2,
		*block_l2 ^ *block_i2);

	NOARR_TUNE_PAR(tmp_layout, noarr::tuning::choice,
		i_vec ^ j_vec,
		j_vec ^ i_vec);

	NOARR_TUNE_PAR(a_layout, noarr::tuning::choice,
		i_vec ^ k_vec,
		k_vec ^ i_vec);
	
	NOARR_TUNE_PAR(b_layout, noarr::tuning::choice,
		k_vec ^ j_vec,
		j_vec ^ k_vec);

	NOARR_TUNE_PAR(c_layout, noarr::tuning::choice,
		j_vec ^ l_vec,
		l_vec ^ j_vec);

	NOARR_TUNE_PAR(d_layout, noarr::tuning::choice,
		i_vec ^ l_vec,
		l_vec ^ i_vec);

	NOARR_TUNE_END();
} tuning;

// initialization function
void init_array(num_t &alpha, num_t &beta, auto A, auto B, auto C, auto D) noexcept {
	// tmp: i x j
	// A: i x k
	// B: k x j
	// C: j x l
	// D: i x l

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	auto ni = A | noarr::get_length<'i'>();
	auto nj = B | noarr::get_length<'j'>();
	auto nk = A | noarr::get_length<'k'>();
	auto nl = C | noarr::get_length<'l'>();

	noarr::traverser(A)
		.for_each([=](auto state) constexpr noexcept {
			auto [i, k] = noarr::get_indices<'i', 'k'>(state);
			A[state] = (num_t)((i * k + 1) % ni) / ni;
		});

	noarr::traverser(B)
		.for_each([=](auto state) constexpr noexcept {
			auto [k, j] = noarr::get_indices<'k', 'j'>(state);
			B[state] = (num_t)(k * (j + 1) % nj) / nj;
		});

	noarr::traverser(C)
		.for_each([=](auto state) constexpr noexcept {
			auto [j, l] = noarr::get_indices<'j', 'l'>(state);
			C[state] = (num_t)((j * (l + 3) + 1) % nl) / nl;
		});

	noarr::traverser(D)
		.for_each([=](auto state) constexpr noexcept {
			auto [i, l] = noarr::get_indices<'i', 'l'>(state);
			D[state] = (num_t)(i * (l + 2) % nk) / nk;
		});
}

// computation kernel
template<class Order1 = noarr::neutral_proto, class Order2 = noarr::neutral_proto>
[[gnu::flatten, gnu::noinline]]
void kernel_2mm(num_t alpha, num_t beta, auto tmp, auto A, auto B, auto C, auto D, Order1 order1 = {}, Order2 order2 = {}) noexcept {
	// tmp: i x j
	// A: i x k
	// B: k x j
	// C: j x l
	// D: i x l

	noarr::planner(tmp, A, B)
		.for_each_elem([alpha](auto &&tmp, auto &&A, auto &&B) constexpr noexcept {
			tmp += alpha * A * B;
		})
		.template for_sections<'i', 'j'>([tmp](auto inner) constexpr noexcept {
			auto state = inner.state();

			tmp[state] = 0;

			inner();
		})
		.order(noarr::hoist<'j'>())
		.order(noarr::hoist<'i'>())
		.order(order1)
		();

	noarr::planner(D, tmp, C)
		.for_each_elem([](auto &&D, auto &&tmp, auto &&C) constexpr noexcept {
			D += tmp * C;
		})
		.template for_sections<'i', 'l'>([D, beta](auto inner) constexpr noexcept {
			auto state = inner.state();

			D[state] *= beta;

			inner();
		})
		.order(noarr::hoist<'l'>())
		.order(noarr::hoist<'i'>())
		.order(order2)
		();
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nj = NJ;
	std::size_t nk = NK;
	std::size_t nl = NL;

	// data
	num_t alpha;
	num_t beta;

	auto set_sizes = noarr::set_length<'i'>(ni) ^ noarr::set_length<'j'>(nj) ^ noarr::set_length<'k'>(nk) ^ noarr::set_length<'l'>(nl);

	auto tmp = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.tmp_layout ^ set_sizes);

	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.a_layout ^ set_sizes);
	auto B = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.b_layout ^ set_sizes);
	auto C = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.c_layout ^ set_sizes);

	auto D = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.d_layout ^ set_sizes);

	// initialize data
	init_array(alpha, beta, A.get_ref(), B.get_ref(), C.get_ref(), D.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_2mm(alpha, beta, tmp.get_ref(), A.get_ref(), B.get_ref(), C.get_ref(), D.get_ref(), *tuning.order1, *tuning.order2);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, D.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << duration.count() << std::endl;
}
