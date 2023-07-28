#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "jacobi-1d.hpp"

using num_t = DATA_TYPE;

namespace {

// initialization function
void init_array(auto A, auto B) {
	// A: i
	// B: i

	auto n = A | noarr::get_length<'i'>();

	noarr::traverser(A, B)
		.template for_each<'i'>([=](auto state) {
			auto i = noarr::get_index<'i'>(state);

			A[state] = ((num_t) i + 2) / n;
			B[state] = ((num_t) i + 3) / n;
		});
}


// computation kernel
void kernel_jacobi_1d(std::size_t steps, auto A, auto B) {
	// A: i
	// B: i

	auto traverser = noarr::traverser(A, B).order(noarr::bcast<'t'>(steps));

	traverser
		.order(noarr::symmetric_span<'i'>(traverser.top_struct(), 1))
		.template for_dims<'t'>([=](auto inner) {
			inner.for_each([=](auto state) {
				B[state] = 0.33333 * (A[neighbor<'i'>(state, -1)] + A[state] + A[neighbor<'i'>(state, +1)]);
			});

			inner.for_each([=](auto state) {
				A[state] = 0.33333 * (B[neighbor<'i'>(state, -1)] + B[state] + B[neighbor<'i'>(state, +1)]);
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
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));
	auto B = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));

	// initialize data
	init_array(A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_jacobi_1d(t, A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, A);
	}

	std::cerr << duration.count() << std::endl;
}
