#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/planner.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "doitgen.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto r_vec =  noarr::vector<'r'>();
constexpr auto q_vec =  noarr::vector<'q'>();
constexpr auto p_vec =  noarr::vector<'p'>();
constexpr auto s_vec =  noarr::vector<'s'>();

struct tuning {
	DEFINE_PROTO_STRUCT(block_r, noarr::hoist<'r'>());
	DEFINE_PROTO_STRUCT(block_q, noarr::hoist<'q'>());

	DEFINE_PROTO_STRUCT(order, block_q ^ block_r);

	DEFINE_PROTO_STRUCT(a_layout, p_vec ^ q_vec ^ r_vec);
	DEFINE_PROTO_STRUCT(c4_layout, p_vec ^ s_vec);
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
[[gnu::flatten, gnu::noinline]]
void kernel_doitgen(auto A, auto C4, auto sum, Order order = {}) {
	// A: r x q x p
	// C4: s x p
	// sum: p
	using namespace noarr;

	auto A_rqs = A ^ rename<'p', 's'>();

	#pragma scop
	planner(A, C4, sum) ^ for_dims<'r', 'q'>([=](auto inner) {
		inner ^ for_dims<'p'>([=](auto inner) {
			sum[inner] = 0;

			inner ^ for_each([=](auto state) {
				sum[state] += A_rqs[state] * C4[state];
			}) | planner_execute();
		}) | planner_execute();

		inner ^ for_dims<'p'>([=](auto inner) {
			A[inner] = sum[inner];
		}) | planner_execute();
	}) ^ order | planner_execute();
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t nr = NR;
	std::size_t nq = NQ;
	std::size_t np = NP;

	auto set_lengths = noarr::set_length<'r'>(nr) ^ noarr::set_length<'q'>(nq) ^ noarr::set_length<'s'>(np) ^ noarr::set_length<'p'>(np);

	// data
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto sum = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'p'>(np));
	auto C4 = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.c4_layout ^ set_lengths);

	// initialize data
	init_array(A.get_ref(), C4.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_doitgen(A.get_ref(), C4.get_ref(), sum.get_ref(), tuning.order);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, A.get_ref() ^ noarr::reorder<'r', 'q', 'p'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
