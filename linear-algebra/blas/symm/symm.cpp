#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures/extra/shortcuts.hpp>
#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "symm.hpp"

using num_t = DATA_TYPE;

namespace {

// initialization function
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: i x j

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	auto ni = C | noarr::get_length<'i'>();
	auto nj = C | noarr::get_length<'j'>();

	noarr::traverser(C)
		.for_each([=](auto state) {
			auto [i, j] = noarr::get_indices<'i', 'j'>(state);
			C[state] = (num_t)((i + j) % 100) / ni;
			B[state] = (num_t)((nj + i - j) % 100) / ni;
		});

	noarr::traverser(A)
		.template for_dims<'i'>([=](auto inner) {
			auto state = inner.state();
			
			inner.order(noarr::slice<'k'>(0, noarr::get_index<'i'>(state) + 1))
				.for_each([=](auto state) {
					auto [i, k] = noarr::get_indices<'i', 'k'>(state);
					A[state] = (num_t)((i + k) % 100) / ni;
				});
			
			inner.order(noarr::shift<'k'>(noarr::get_index<'i'>(state) + 1))
				.for_each([=](auto state) {
					A[state] = -999;
				});
		});
}

// computation kernel
void kernel_symm(num_t alpha, num_t beta, auto C, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: i x j

	auto C_renamed = C ^ noarr::rename<'i', 'k'>();
	auto B_renamed = B ^ noarr::rename<'i', 'k'>();

	noarr::traverser(C, A, B)
		.template for_dims<'i', 'j'>([=](auto inner) {
			num_t temp = 0;
			auto state = inner.state();

			inner
				.order(noarr::slice<'k'>(0, noarr::get_index<'i'>(state)))
				. template for_each<'k'>([=, &temp](auto state) {
					C_renamed[state] += alpha * B[state] * A[state];
					temp += B_renamed[state] * A[state];
				});

			C[state] = beta * C[state] + alpha * B[state] * A[state & noarr::idx<'k'>(noarr::get_index<'i'>(state))] + alpha * temp; // TODO: A_diag
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
	num_t beta;

	auto C = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(ni, nj));
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'k'>(ni, ni));
	auto B = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(ni, nj));

	// initialize data
	init_array(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_symm(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	// print results
	if (argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, C.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << duration << std::endl;
}
