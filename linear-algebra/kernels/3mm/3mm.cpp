#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/planner.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "3mm.hpp"

// autotuning
#include <noarr/structures/tuning/formatters/opentuner_formatter.hpp>

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec =  noarr::vector<'i'>();
constexpr auto j_vec =  noarr::vector<'j'>();
constexpr auto k_vec =  noarr::vector<'k'>();
constexpr auto l_vec =  noarr::vector<'l'>();
constexpr auto m_vec =  noarr::vector<'m'>();

struct tuning {
	NOARR_TUNE_BEGIN(noarr::tuning::opentuner_formatter(
		std::cout,
		std::make_shared<noarr::tuning::cmake_compile_command_builder>("../..", "build", "3mm", "-D" STRINGIFY(DATASET_SIZE) " -D" STRINGIFY(DATA_TYPE_CHOICE)),
		std::make_shared<noarr::tuning::direct_run_command_builder>("build/3mm"),
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

	NOARR_TUNE_PAR(block_j2, noarr::tuning::choice,
		noarr::bcast<'J'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_l2, noarr::tuning::choice,
		noarr::bcast<'L'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'l', 'L', 'l', 'u'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'l', 'L', 'l', 'u'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'l', 'L', 'l', 'u'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'l', 'L', 'l', 'u'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'l', 'L', 'l', 'u'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'l', 'L', 'l', 'u'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_i3, noarr::tuning::choice,
		noarr::bcast<'I'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_l3, noarr::tuning::choice,
		noarr::bcast<'L'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'l', 'L', 'l', 'u'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'l', 'L', 'l', 'u'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'l', 'L', 'l', 'u'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'l', 'L', 'l', 'u'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'l', 'L', 'l', 'u'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'l', 'L', 'l', 'u'>(noarr::lit<64>));

	NOARR_TUNE_PAR(order1, noarr::tuning::choice,
		noarr::hoist<'i'>() ^ *block_j1 ^ *block_i1,
		noarr::hoist<'j'>() ^ *block_i1 ^ *block_j1);

	NOARR_TUNE_PAR(order2, noarr::tuning::choice,
		noarr::hoist<'j'>() ^ *block_l2 ^ *block_j2,
		noarr::hoist<'l'>() ^ *block_j2 ^ *block_l2);
	
	NOARR_TUNE_PAR(order3, noarr::tuning::choice,
		noarr::hoist<'i'>() ^ *block_l3 ^ *block_i3,
		noarr::hoist<'l'>() ^ *block_i3 ^ *block_l3);

	NOARR_TUNE_PAR(e_layout, noarr::tuning::choice,
		j_vec ^ i_vec);

	NOARR_TUNE_PAR(a_layout, noarr::tuning::choice,
		k_vec ^ i_vec);

	NOARR_TUNE_PAR(b_layout, noarr::tuning::choice,
		j_vec ^ k_vec);

	NOARR_TUNE_PAR(f_layout, noarr::tuning::choice,
		l_vec ^ j_vec);

	NOARR_TUNE_PAR(c_layout, noarr::tuning::choice,
		m_vec ^ j_vec);

	NOARR_TUNE_PAR(d_layout, noarr::tuning::choice,
		l_vec ^ m_vec);
	
	NOARR_TUNE_PAR(g_layout, noarr::tuning::choice,
		l_vec ^ i_vec);

	NOARR_TUNE_END();
} tuning;

// initialization function
void init_array(auto A, auto B, auto C, auto D) noexcept {
	// A: i x k
	// B: k x j
	// C: j x m
	// D: m x l

	auto ni = A | noarr::get_length<'i'>();
	auto nj = B | noarr::get_length<'j'>();
	auto nk = A | noarr::get_length<'k'>();
	auto nl = D | noarr::get_length<'l'>();

	noarr::traverser(A)
		.for_each([=](auto state) {
			auto [i, k] = noarr::get_indices<'i', 'k'>(state);
			A[state] = (num_t)((i * k + 1) % ni) / (5 * ni);
		});

	noarr::traverser(B)
		.for_each([=](auto state) {
			auto [k, j] = noarr::get_indices<'k', 'j'>(state);
			B[state] = (num_t)((k * (j + 1) + 2) % nj) / (5 * nj);
		});

	noarr::traverser(C)
		.for_each([=](auto state) {
			auto [j, m] = noarr::get_indices<'j', 'm'>(state);
			C[state] = (num_t)(j * (m + 3) % nl) / (5 * nl);
		});

	noarr::traverser(D)
		.for_each([=](auto state) {
			auto [m, l] = noarr::get_indices<'m', 'l'>(state);
			D[state] = (num_t)((m * (l + 2) + 2) % nk) / (5 * nk);
		});
}

// computation kernel
template<class Order1 = noarr::neutral_proto, class Order2 = noarr::neutral_proto, class Order3 = noarr::neutral_proto>
[[gnu::flatten, gnu::noinline]]
void kernel_3mm(auto E, auto A, auto B, auto F, auto C, auto D, auto G, Order1 order1 = {}, Order2 order2 = {}, Order3 order3 = {}) noexcept {
	// E: i x j
	// A: i x k
	// B: k x j
	// F: j x l
	// C: j x m
	// D: m x l
	// G: i x l

	constexpr auto madd = [](auto &&m, auto &&l, auto &&r) {
		m += l * r;
	};

	noarr::planner(E, A, B)
		.for_each_elem(madd)
		.template for_sections<'i', 'j'>([=](auto inner) {
			E[inner.state()] = 0;
			inner();
		})
		.order(noarr::hoist<'k'>())
		.order(noarr::hoist<'j'>())
		.order(noarr::hoist<'i'>())
		.order(order1)
		();

	noarr::planner(F, C, D)
		.for_each_elem(madd)
		.template for_sections<'j', 'l'>([=](auto inner) {
			F[inner.state()] = 0;
			inner();
		})
		.order(noarr::hoist<'m'>())
		.order(noarr::hoist<'l'>())
		.order(noarr::hoist<'j'>())
		.order(order2)
		();

	noarr::planner(G, E, F)
		.for_each_elem(madd)
		.template for_sections<'i', 'l'>([=](auto inner) {
			G[inner.state()] = 0;
			inner();
		})
		.order(noarr::hoist<'j'>())
		.order(noarr::hoist<'l'>())
		.order(noarr::hoist<'i'>())
		.order(order3)
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
	std::size_t nm = NM;

	auto set_sizes = noarr::set_length<'i'>(ni) ^ noarr::set_length<'j'>(nj) ^ noarr::set_length<'k'>(nk) ^ noarr::set_length<'l'>(nl) ^ noarr::set_length<'m'>(nm);

	// data
	auto E = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.e_layout ^ set_sizes);
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.a_layout ^ set_sizes);
	auto B = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.b_layout ^ set_sizes);

	auto F = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.f_layout ^ set_sizes);
	auto C = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.c_layout ^ set_sizes);
	auto D = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.d_layout ^ set_sizes);

	auto G = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.g_layout ^ set_sizes);

	// initialize data
	init_array(A.get_ref(), B.get_ref(), C.get_ref(), D.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_3mm(E.get_ref(), A.get_ref(), B.get_ref(),
		F.get_ref(), C.get_ref(), D.get_ref(),
		G.get_ref(),
		*tuning.order1, *tuning.order2, *tuning.order3);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, G.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << duration.count() << std::endl;
}
