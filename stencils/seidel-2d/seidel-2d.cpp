#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "seidel-2d.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec =  noarr::vector<'i'>();
constexpr auto j_vec =  noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto A) noexcept {
	// A: i x j

	auto n = A | noarr::get_length<'i'>();

	noarr::traverser(A)
		.for_each([=](auto state) {
			auto [i, j] = noarr::get_indices<'i', 'j'>(state);

			A[state] = ((num_t)i * (j + 2) + 2) / n;
		});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_seidel_2d(std::size_t steps, auto A) noexcept {
	// A: i x j

	auto traverser = noarr::traverser(A).order(noarr::bcast<'t'>(steps));

	#pragma scop
	traverser
		.order(noarr::symmetric_spans<'i', 'j'>(traverser.top_struct(), 1, 1))
		.order(noarr::reorder<'t', 'i', 'j'>())
		.for_each([=](auto state) {
			A[state] = (
				A[state - noarr::idx<'i'>(1) - noarr::idx<'j'>(1)] + // corner
				A[state - noarr::idx<'i'>(1)] +          // edge
				A[state - noarr::idx<'i'>(1) + noarr::idx<'j'>(1)] + // corner
				A[state - noarr::idx<'j'>(1)] +          // edge
				A[state] +                             // center
				A[state + noarr::idx<'j'>(1)] +          // edge
				A[state + noarr::idx<'i'>(1) - noarr::idx<'j'>(1)] + // corner
				A[state + noarr::idx<'i'>(1)] +          // edge
				A[state + noarr::idx<'i'>(1) + noarr::idx<'j'>(1)]) / (num_t)9.0; // corner
		});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;
	std::size_t t = TSTEPS;

	// data
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));

	// initialize data
	init_array(A.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_seidel_2d(t, A.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, A.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
