#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/planner.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "atax.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec =  noarr::vector<'i'>();
constexpr auto j_vec =  noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto A, auto x) noexcept {
	// A: i x j
	// x: j

	auto ni = A | noarr::get_length<'i'>();
	auto nj = A | noarr::get_length<'j'>();

	noarr::traverser(x).for_each([=](auto state) {
		auto j = noarr::get_index<'j'>(state);
		x[state] = 1 + j / (num_t)nj;
	});

	noarr::traverser(A).for_each([=](auto state) {
		auto [i, j] = noarr::get_indices<'i', 'j'>(state);
		A[state] = (num_t)((i + j) % nj) / (5 * ni);
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_atax(auto A, auto x, auto y, auto tmp) noexcept {
	// A: i x j
	// x: j
	// y: j
	// tmp: i

	#pragma scop
	noarr::traverser(y).for_each([=](auto state) {
		y[state] = 0;
	});

	noarr::traverser(tmp, A, x, y).template for_dims<'i'>([=](auto inner) {
		tmp[inner.state()] = 0;

		inner.for_each([=](auto state) {
			tmp[state] += A[state] * x[state];
		});

		inner.for_each([=](auto state) {
			y[state] += A[state] * tmp[state];
		});
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nj = NJ;

	// data
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.c_layout ^ noarr::set_length<'i'>(ni) ^ noarr::set_length<'j'>(nj));

	auto x = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'j'>(nj));
	auto y = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'j'>(nj));

	auto tmp = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(ni));

	// initialize data
	init_array(A.get_ref(), x.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_atax(A.get_ref(), x.get_ref(), y.get_ref(), tmp.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, y);
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
