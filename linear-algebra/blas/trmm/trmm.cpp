#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/planner.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "trmm.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec =  noarr::vector<'i'>();
constexpr auto j_vec =  noarr::vector<'j'>();
constexpr auto k_vec =  noarr::vector<'k'>();

struct tuning {
	DEFINE_PROTO_STRUCT(block_i, noarr::hoist<'i'>());
	DEFINE_PROTO_STRUCT(block_j, noarr::hoist<'j'>());

	DEFINE_PROTO_STRUCT(order, block_j ^ block_i);

	DEFINE_PROTO_STRUCT(a_layout, i_vec ^ k_vec);
	DEFINE_PROTO_STRUCT(b_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(num_t &alpha, auto A, auto B) {
	// A: k x i
	// B: i x j

	alpha = (num_t)1.5;

	auto ni = A | noarr::get_length<'i'>();
	auto nj = B | noarr::get_length<'j'>();

	noarr::traverser(A)
		.template for_dims<'k'>([=](auto inner) {
			auto state = inner.state();

			auto k = noarr::get_index<'k'>(state);

			inner.order(noarr::slice<'i'>(0, k))
				.for_each([=](auto state) {
					auto i = noarr::get_index<'i'>(state);
					A[state] = (num_t)((k + i) % ni) / ni;
				});

			A[state & noarr::idx<'i'>(k)] = 1.0;
		});

	noarr::traverser(B).for_each([=](auto state) {
		auto [i, j] = noarr::get_indices<'i', 'j'>(state);

		B[state] = (num_t)((nj + (i - j)) % nj) / nj;
	});
}

// computation kernel
template<class Order = noarr::neutral_proto>
[[gnu::flatten, gnu::noinline]]
void kernel_trmm(num_t alpha, auto A, auto B, Order order = {}) {
	// A: k x i
	// B: i x j

	auto B_renamed = B ^ noarr::rename<'i', 'k'>();

	#pragma scop
	noarr::planner(A, B, B_renamed)
		.for_each_elem([](auto &&A, auto &&B, auto &&B_renamed) {
			B += A * B_renamed;
		})
		.template for_sections<'i', 'j'>([=](auto inner) {
			auto state = inner.state();

			inner
				.order(noarr::shift<'k'>(noarr::get_index<'i'>(state) + 1))
				();

			B[state] *= alpha;
		})
		.order(noarr::hoist<'j'>())
		.order(noarr::hoist<'i'>())
		.order(order)
		();
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nj = NJ;

	// data
	num_t alpha;

	auto set_lengths = noarr::set_length<'i'>(ni) ^ noarr::set_length<'k'>(ni) ^ noarr::set_length<'j'>(nj);

	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto B = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize data
	init_array(alpha, A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_trmm(alpha, A.get_ref(), B.get_ref(), tuning.order);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, B.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
