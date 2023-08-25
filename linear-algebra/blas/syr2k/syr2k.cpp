#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "syr2k.hpp"

using num_t = DATA_TYPE;

namespace {

// initialization function
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: i x k

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	auto ni = C | noarr::get_length<'i'>();
	auto nk = A | noarr::get_length<'k'>();

	noarr::traverser(A, B)
		.for_each([=](auto state) {
			auto [i, k] = noarr::get_indices<'i', 'k'>(state);
			A[state] = (num_t)((i * k + 1) % ni) / ni;
			B[state] = (num_t)((i * k + 2) % nk) / nk;
		});

	noarr::traverser(C)
		.for_each([=](auto state) {
			auto [i, j] = noarr::get_indices<'i', 'j'>(state);
			C[state] = (num_t)((i * j + 3) % ni) / nk;
		});
}

// computation kernel
void kernel_syr2k(num_t alpha, num_t beta, auto C, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: i x k

	auto A_renamed = A ^ noarr::rename<'i', 'j'>();
	auto B_renamed = B ^ noarr::rename<'i', 'j'>();

	noarr::traverser(C, A, B)
		.template for_dims<'i'>(
			[=](auto inner) {
				auto state = inner.state();

				inner
					.order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state) + 1))
					.template for_each<'j'>([=](auto state) {
						C[state] *= beta;
					});

				inner
					.order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state) + 1))
					.for_each([=](auto state) {
						C[state] += A_renamed[state] * alpha * B[state] + B_renamed[state] * alpha * A[state];
					});
			});
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nk = NK;

	// data
	num_t alpha;
	num_t beta;
	
	auto C = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(ni, ni));
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'k'>(ni, nk));
	auto B = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'k'>(ni, nk));

	// initialize data
	init_array(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_syr2k(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, C.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << duration.count() << std::endl;
}
