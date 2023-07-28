#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "jacobi-2d.hpp"

using num_t = DATA_TYPE;

namespace {

// initialization function
void init_array(auto A, auto B) {
	// A: i x j
	// B: i x j

	auto n = A | noarr::get_length<'i'>();

	noarr::traverser(A, B)
		.for_each([=](auto state) {
			auto [i, j] = noarr::get_indices<'i', 'j'>(state);

			A[state] = ((num_t)i * (j + 2) + 2) / n;
			B[state] = ((num_t)i * (j + 3) + 3) / n;
		});
}


// computation kernel
void kernel_jacobi_2d(std::size_t steps, auto A, auto B) {
	// A: i x j
	// B: i x j

	auto traverser = noarr::traverser(A, B).order(noarr::bcast<'t'>(steps));

	traverser
		.order(noarr::symmetric_spans<'i', 'j'>(traverser.top_struct(), 1, 1))
		.template for_dims<'t'>([=](auto inner) {
			inner.for_each([=](auto state) {
				B[state] = (num_t).2 * (
					A[state] +
					A[neighbor<'j'>(state, -1)] +
					A[neighbor<'j'>(state, +1)] +
					A[neighbor<'i'>(state, +1)] +
					A[neighbor<'i'>(state, -1)]);
			});

			inner.for_each([=](auto state) {
				A[state] = (num_t).2 * (
					B[state] +
					B[neighbor<'j'>(state, -1)] +
					B[neighbor<'j'>(state, +1)] +
					B[neighbor<'i'>(state, +1)] +
					B[neighbor<'i'>(state, -1)]);
			});
		});
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;
	std::size_t t = TSTEPS;

	// data
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(n, n));
	auto B = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(n, n));

	// initialize data
	init_array(A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_jacobi_2d(t, A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, A.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << duration.count() << std::endl;
}
