#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "gemm.hpp"

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

	DEFINE_PROTO_STRUCT(c_layout, i_vec ^ j_vec);
	DEFINE_PROTO_STRUCT(a_layout, i_vec ^ k_vec);
	DEFINE_PROTO_STRUCT(b_layout, k_vec ^ j_vec);
} tuning;

// initialization function
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) noexcept {
	// C: i x j
	// A: i x k
	// B: k x j

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	noarr::traverser(C)
		.for_each([=](auto state) constexpr noexcept {
			auto [i, j] = noarr::get_indices<'i', 'j'>(state);
			C[state] = (num_t)((i * j + 1) % (C | noarr::get_length<'i'>())) / (C | noarr::get_length<'i'>());
		});

	noarr::traverser(A)
		.for_each([=](auto state) constexpr noexcept {
			auto [i, k] = noarr::get_indices<'i', 'k'>(state);
			A[state] = (num_t)(i * (k + 1) % (A | noarr::get_length<'k'>())) / (A | noarr::get_length<'k'>());
		});

	noarr::traverser(B)
		.for_each([=](auto state) constexpr noexcept {
			auto [k, j] = noarr::get_indices<'k', 'j'>(state);
			B[state] = (num_t)(k * (j + 2) % (B | noarr::get_length<'j'>())) / (B | noarr::get_length<'j'>());
		});
}

// computation kernel
template<class Order = noarr::neutral_proto>
[[gnu::flatten, gnu::noinline]]
void kernel_gemm(num_t alpha, num_t beta, auto C, auto A, auto B, Order order = {}) noexcept {
	// C: i x j
	// A: i x k
	// B: k x j

	noarr::traverser(C)
		.for_each([=](auto state) constexpr noexcept {
			C[state] *= beta;
		});

	noarr::traverser(C, A, B).order(order)
		.for_each([=](auto state) constexpr noexcept {
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

	auto set_lengths = noarr::set_length<'i'>(ni) ^ noarr::set_length<'j'>(nj) ^ noarr::set_length<'k'>(nk);

	auto C = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.c_layout ^ set_lengths);
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto B = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize data
	init_array(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_gemm(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref(), tuning.order);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, C.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << duration.count() << std::endl;
}
