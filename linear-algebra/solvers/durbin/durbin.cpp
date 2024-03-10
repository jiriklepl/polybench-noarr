#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "durbin.hpp"

using num_t = DATA_TYPE;

namespace {

// initialization function
void init_array(auto r) {
	// r: i

	auto n = r | noarr::get_length<'i'>();

	noarr::traverser(r)
		.for_each([=](auto state) {
			auto i = noarr::get_index<'i'>(state);
			r[state] = n + 1 - i;
		});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_durbin(auto r, auto y) {
	// r: i
	// y: i

	auto z = noarr::make_bag(r.structure());

	auto r_k = r ^ noarr::rename<'i', 'k'>();
	auto y_k = y ^ noarr::rename<'i', 'k'>();

	num_t alpha;
	num_t beta;
	num_t sum;

	#pragma scop
	y[noarr::idx<'i'>(0)] = -r[noarr::idx<'i'>(0)];
	beta = 1;
	alpha = -r[noarr::idx<'i'>(0)];

	noarr::traverser(r, y, r_k, y_k)
		.order(noarr::shift<'k'>(1))
		.template for_dims<'k'>([=, &alpha, &beta, &sum, z = z.get_ref()](auto inner) {
			auto state = inner.state();

			beta = (1 - alpha * alpha) * beta;
			sum = 0;

			auto traverser = inner
				.order(noarr::slice<'i'>(0, noarr::get_index<'k'>(state)));

			traverser.for_each([=, &sum](auto state) {
				auto [i, k] = noarr::get_indices<'i', 'k'>(state);
				// sum += r_k[noarr::neighbor<'k'>(state, -i - 1)] * y[state];
				sum += r[noarr::idx<'i'>(k - i - 1)] * y[state];
			});

			alpha = -(r_k[state] + sum) / beta;

			traverser.for_each([=, &alpha](auto state) {
				auto [i, k] = noarr::get_indices<'i', 'k'>(state);
				// z[state] = y[state] + alpha * y_k[noarr::neighbor<'k'>(state, -i - 1)];
				z[state] = y[state] + alpha * y[noarr::idx<'i'>(k - i - 1)];
			});

			traverser.for_each([=](auto state) {
				y[state] = z[state];
			});

			y_k[state] = alpha;
		});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// data
	auto r = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));
	auto y = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));

	// initialize data
	init_array(r.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_durbin(r.get_ref(), y.get_ref());

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
