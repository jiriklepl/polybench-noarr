#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "gesummv.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec =  noarr::vector<'i'>();
constexpr auto j_vec =  noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(block_i, noarr::neutral_proto());
	DEFINE_PROTO_STRUCT(block_j, noarr::neutral_proto());

	DEFINE_PROTO_STRUCT(order, block_i ^ block_j);

	DEFINE_PROTO_STRUCT(a_layout, i_vec ^ j_vec);
	DEFINE_PROTO_STRUCT(b_layout, i_vec ^ j_vec);
} tuning;

// initialization function
void init_array(num_t &alpha, num_t &beta, auto A, auto B, auto x) noexcept {
	// A: i x j
	// B: i x j
	// x: j

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	auto n = A | noarr::get_length<'i'>();

	noarr::traverser(A, B, x)
		.template for_dims<'i'>([=](auto inner) constexpr noexcept {
			auto state = inner.state();

			auto i = noarr::get_index<'i'>(state);

			x[noarr::idx<'j'>(i)] = (num_t)(i % n) / n;

			inner.for_each([=](auto state) constexpr noexcept {
				auto j = noarr::get_index<'j'>(state);

				A[state] = (num_t)((i * j + 1) % n) / n;
				B[state] = (num_t)((i * j + 2) % n) / n;
			});
		});
}

// computation kernel
template<class Order = noarr::neutral_proto>
[[gnu::flatten, gnu::noinline]]
void kernel_gesummv(num_t alpha, num_t beta, auto A, auto B, auto tmp, auto x, auto y, Order order = {}) noexcept {
	// A: i x j
	// B: i x j
	// tmp: i
	// x: j
	// y: i

	#pragma scop
	noarr::traverser(A, B, tmp, x, y).for_each([=](auto state) constexpr noexcept {
		tmp[state] = 0;
		y[state] = 0;
	});

	noarr::traverser(A, B, tmp, x, y)
		.order(order)
		.for_each([=](auto state) constexpr noexcept {
			tmp[state] += A[state] * x[state];
			y[state] += B[state] * x[state];
		});

	noarr::traverser(y, tmp)
		.for_each([=](auto state) constexpr noexcept {
			y[state] = alpha * tmp[state] + beta * y[state];
		});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// data
	num_t alpha;
	num_t beta;

	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));
	auto B = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));
	auto tmp = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));
	auto x = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'j'>(n));
	auto y = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));

	// initialize data
	init_array(alpha, beta, A.get_ref(), B.get_ref(), x.get_ref());


	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_gesummv(alpha, beta, A.get_ref(), B.get_ref(), tmp.get_ref(), x.get_ref(), y.get_ref(), tuning.order);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, y);
	}

	std::cerr << duration.count() << std::endl;
}
