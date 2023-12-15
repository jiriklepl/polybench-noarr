#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "heat-3d.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec =  noarr::vector<'i'>();
constexpr auto j_vec =  noarr::vector<'j'>();
constexpr auto k_vec =  noarr::vector<'k'>();

struct tuning {
	DEFINE_PROTO_STRUCT(block_i, noarr::neutral_proto());
	DEFINE_PROTO_STRUCT(block_j, noarr::neutral_proto());
	DEFINE_PROTO_STRUCT(block_k, noarr::neutral_proto());

	DEFINE_PROTO_STRUCT(order, block_i ^ block_j ^ block_k);

	DEFINE_PROTO_STRUCT(a_layout, i_vec ^ j_vec ^ k_vec);
	DEFINE_PROTO_STRUCT(b_layout, i_vec ^ j_vec ^ k_vec);
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

	#pragma scop
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
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;
	std::size_t t = TSTEPS;

	auto set_lengths = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n) ^ noarr::set_length<'k'>(n);

	// data
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto B = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize data
	init_array(A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_heat_3d(t, A.get_ref(), B.get_ref(), tuning.order);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, A.get_ref() ^ noarr::reorder<'i', 'j', 'k'>());
	}

	std::cerr << duration.count() << std::endl;
}
