#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "trmm.hpp"

using num_t = DATA_TYPE;

namespace {

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
				.template for_each<'i'>([=](auto state) {
					auto i = noarr::get_index<'i'>(state);
					A[state] = (num_t)((k + i) % ni) / ni;
				});
			
			A[state & noarr::idx<'i'>(k)] = 1.0;
		});

	noarr::traverser(B)
		.template for_each([=](auto state) {

			auto [i, j] = noarr::get_indices<'i', 'j'>(state);

			B[state] = (num_t)((nj + (i - j)) % nj) / nj;
		});
}

// computation kernel
void kernel_trmm(num_t alpha, auto A, auto B) {
	// A: k x i
	// B: i x j

	auto B_renamed = B ^ noarr::rename<'i', 'k'>();

	noarr::traverser(A, B)
		.template for_dims<'i', 'j'>([=](auto inner) {
			auto state = inner.state();

			inner
				.order(noarr::shift<'k'>(noarr::get_index<'i'>(state) + 1))
				.template for_each<'k'>([=](auto state) {
					B[state] += A[state] * B_renamed[state];
				});
			
			B[state] *= alpha;
		});
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nj = NJ;

	// data
	num_t alpha;

	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'k', 'i'>(ni, ni));
	auto B = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(ni, nj));

	// initialize data
	init_array(alpha, A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_trmm(alpha, A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, B.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << duration.count() << std::endl;
}
