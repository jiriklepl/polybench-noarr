#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "gramschmidt.hpp"

using num_t = DATA_TYPE;

namespace {

// initialization function
void init_array(auto A, auto R, auto Q) noexcept {
	// A: i x k
	// R: k x j
	// Q: i x k

	auto ni = A | noarr::get_length<'i'>();

	noarr::traverser(A, Q)
		.for_each([=](auto state) constexpr noexcept {
			auto i = noarr::get_index<'i'>(state);
			auto k = noarr::get_index<'k'>(state);

			A[state] =(((num_t)((i * k) % ni) / ni) * 100) + 10;
			Q[state] = 0.0;
		});

	noarr::traverser(R)
		.for_each([=](auto state) constexpr noexcept {
			R[state] = 0.0;
		});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_gramschmidt(auto A, auto R, auto Q) noexcept {
	// A: i x k
	// R: k x j
	// Q: i x k

	auto A_ij = A ^ noarr::rename<'k', 'j'>();

	#pragma scop
	noarr::traverser(A_ij, R, Q)
		.template for_dims<'k'>([=](auto inner) constexpr noexcept {
			auto state = inner.state();
			num_t norm = 0;

			inner.template for_each<'i'>([=, &norm](auto state) constexpr noexcept {
				norm += A[state] * A[state];
			});

			auto R_diag = R ^ noarr::fix<'j'>(noarr::get_index<'k'>(state));

			R_diag[state] = std::sqrt(norm);

			inner.template for_each<'i'>([=](auto state) constexpr noexcept {
				Q[state] = A[state] / R_diag[state];
			});

			inner
				.order(noarr::shift<'j'>(noarr::get_index<'k'>(state) + 1))
				.template for_dims<'j'>([=](auto inner) constexpr noexcept {
					auto state = inner.state();

					R[state] = 0;

					inner.for_each([=](auto state) constexpr noexcept {
						R[state] = R[state] + Q[state] * A_ij[state];
					});

					inner.for_each([=](auto state) constexpr noexcept {
						A_ij[state] = A_ij[state] - Q[state] * R[state];
					});
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
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'k'>(ni, nj));
	auto R = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'k', 'j'>(nj, nj));
	auto Q = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'k'>(ni, nj));

	// initialize data
	init_array(A.get_ref(), R.get_ref(), Q.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_gramschmidt(A.get_ref(), R.get_ref(), Q.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, R.get_ref() ^ noarr::hoist<'k'>());
		noarr::serialize_data(std::cout, Q.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << duration.count() << std::endl;
}
