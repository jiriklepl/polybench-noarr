#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures/extra/shortcuts.hpp>
#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "2mm.hpp"

using num_t = DATA_TYPE;

namespace {

// initialization function
void init_array(num_t &alpha, num_t &beta, auto A, auto B, auto C, auto D) {
	// tmp: i x j
	// A: i x k
	// B: k x j
	// C: j x l
	// D: i x l

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	auto ni = A | noarr::get_length<'i'>();
	auto nj = B | noarr::get_length<'j'>();
	auto nk = A | noarr::get_length<'k'>();
	auto nl = C | noarr::get_length<'l'>();

	noarr::traverser(A)
		.for_each([=](auto state) {
			auto [i, k] = noarr::get_indices<'i', 'k'>(state);
			A[state] = (num_t)((i * k + 1) % ni) / ni;
		});

	noarr::traverser(B)
		.for_each([=](auto state) {
			auto [k, j] = noarr::get_indices<'k', 'j'>(state);
			B[state] = (num_t)(k * (j + 1) % nj) / nj;
		});

	noarr::traverser(C)
		.for_each([=](auto state) {
			auto [j, l] = noarr::get_indices<'j', 'l'>(state);
			C[state] = (num_t)((j * (l + 3) + 1) % nl) / nl;
		});

	noarr::traverser(D)
		.for_each([=](auto state) {
			auto [i, l] = noarr::get_indices<'i', 'l'>(state);
			D[state] = (num_t)(i * (l + 2) % nk) / nk;
		});
}

// computation kernel
void kernel_2mm(num_t alpha, num_t beta, auto tmp, auto A, auto B, auto C, auto D) {
	// tmp: i x j
	// A: i x k
	// B: k x j
	// C: j x l
	// D: i x l

	noarr::traverser(tmp, A, B)
		.template for_dims<'i', 'j'>([=](auto inner) {
			auto state = inner.state();

			tmp[state] = 0;

			inner.template for_each<'k'>([=](auto state) {
				tmp[state] += alpha * A[state] * B[state];
			});
		});

	noarr::traverser(C, D, tmp)
		.template for_dims<'i', 'l'>([=](auto inner) {
			auto state = inner.state();

			D[state] *= beta;

			inner.template for_each<'j'>([=](auto state) {
				D[state] += tmp[state] * C[state];
			});
		});
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nj = NJ;
	std::size_t nk = NK;
	std::size_t nl = NL;

	// data
	num_t alpha;
	num_t beta;

	auto tmp = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(ni, nj));

	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'k'>(ni, nk));
	auto B = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'k', 'j'>(nk, nj));
	auto C = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'j', 'l'>(nj, nl));

	auto D = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'l'>(ni, nl));

	// initialize data
	init_array(alpha, beta, A.get_ref(), B.get_ref(), C.get_ref(), D.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_2mm(alpha, beta, tmp.get_ref(), A.get_ref(), B.get_ref(), C.get_ref(), D.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	// print results
	if (argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, D.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << duration << std::endl;
}
