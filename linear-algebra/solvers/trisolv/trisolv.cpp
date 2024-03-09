#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "trisolv.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec =  noarr::vector<'i'>();
constexpr auto j_vec =  noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(l_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto L, auto x, auto b) noexcept {
	// L: i x j
	// x: i
	// b: i

	auto n = L | noarr::get_length<'i'>();

	noarr::traverser(L, x, b)
		.template for_dims<'i'>([=](auto inner) {
			auto state = inner.state();
			auto i = noarr::get_index<'i'>(state);

			x[state] = -999;
			b[state] = i;

			inner
				.order(noarr::slice<'j'>(0, i + 1))
				.template for_each<'j'>([=](auto state) {
					auto j = noarr::get_index<'j'>(state);
					L[state] = (num_t)(i + n - j + 1) * 2 / n;
				});
		});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_trisolv(auto L, auto x, auto b) noexcept {
	// L: i x j
	// x: i
	// b: i

	auto x_j = x ^ noarr::rename<'i', 'j'>();

	#pragma scop
	noarr::traverser(L, x, b)
		.template for_dims<'i'>([=](auto inner) {
			auto state = inner.state();

			x[state] = b[state];

			inner
				.order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state)))
				.for_each([=](auto state) {
					x[state] -= L[state] * x_j[state];
				});

			x[state] = x[state] / L[state & noarr::idx<'j'>(noarr::get_index<'i'>(state))];
		});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// data
	auto L = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.l_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));
	auto x = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));
	auto b = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));

	// initialize data
	init_array(L.get_ref(), x.get_ref(), b.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_trisolv(L.get_ref(), x.get_ref(), b.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, x);
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
