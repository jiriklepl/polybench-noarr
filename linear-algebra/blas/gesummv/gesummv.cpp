#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures/extra/shortcuts.hpp>
#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "gesummv.hpp"

using num_t = DATA_TYPE;

namespace {

// initialization function
void init_array(num_t &alpha, num_t &beta, auto A, auto B, auto x) {
	// A: i x j
	// B: i x j
	// x: j

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	auto n = A | noarr::get_length<'i'>();

	noarr::traverser(A, B, x)
		.template for_dims<'i'>([=](auto inner) {
			auto state = inner.state();

			auto i = noarr::get_index<'i'>(state);

			x[noarr::idx<'j'>(i)] = (num_t)(i % n) / n;

			inner.template for_each<'j'>([=](auto state) {
				auto j = noarr::get_index<'j'>(state);

				A[state] = (num_t)((i * j + 1) % n) / n;
				B[state] = (num_t)((i * j + 2) % n) / n;
			});
		});
}

// computation kernel
void kernel_gesummv(num_t alpha, num_t beta, auto A, auto B, auto tmp, auto x, auto y) {
	// A: i x j
	// B: i x j
	// tmp: i
	// x: j
	// y: i

	noarr::traverser(A, B, tmp, x, y)
		.template for_dims<'i'>([=](auto inner) {
			auto state = inner.state();

			tmp[state] = 0;
			y[state] = 0;

			inner.template for_each<'j'>([=](auto state) {
				tmp[state] += A[state] * x[state];
				y[state] += B[state] * x[state];
			});

			y[state] = alpha * tmp[state] + beta * y[state];
		});
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// data
	num_t alpha;
	num_t beta;

	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(n, n));
	auto B = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(n, n));
	auto tmp = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));
	auto x = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'j'>(n));
	auto y = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));

	// initialize data
	init_array(alpha, beta, A.get_ref(), B.get_ref(), x.get_ref());


	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_gesummv(alpha, beta, A.get_ref(), B.get_ref(), tmp.get_ref(), x.get_ref(), y.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, y);
	}

	std::cerr << duration.count() << std::endl;
}
