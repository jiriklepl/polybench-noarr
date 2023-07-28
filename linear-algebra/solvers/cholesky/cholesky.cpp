#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "cholesky.hpp"

using num_t = DATA_TYPE;

namespace {

// initialization function
void init_array(auto A) {
	// A: i x j

	int n = A | noarr::get_length<'i'>();

	noarr::traverser(A)
		.template for_dims<'i'>([=](auto inner) {
			auto state = inner.state();

			auto i = noarr::get_index<'i'>(state);

			auto A_ii = A ^ noarr::fix<'j'>(i);

			inner
				.order(noarr::slice<'j'>(0, i + 1))
				.for_each([=](auto state) {
					A[state] = (num_t) (-(int)noarr::get_index<'j'>(state) % n) / n + 1;
				});

			inner
				.order(noarr::shift<'j'>(i + 1))
				.template for_each<'j'>([=](auto state) {
					A[state] = 0;
				});

			A_ii[state] = 1;
		});

	// make A positive semi-definite
	auto B = noarr::make_bag(A.structure());
	auto B_ref = B.get_ref();

	auto A_ik = A ^ noarr::rename<'j', 'k'>();
	auto A_jk = A ^ noarr::rename<'i', 'j', 'j', 'k'>();

	noarr::traverser(B_ref)
		.for_each([=](auto state) {
			B_ref[state] = 0;
		});

	noarr::traverser(B_ref, A_ik, A_jk)
		.for_each([=](auto state) {
			B_ref[state] += A_ik[state] * A_jk[state];
		});

	noarr::traverser(A, B_ref)
		.template for_each([=](auto state) {
			A[state] = B_ref[state];
		});
}

// computation kernel
void kernel_cholesky(auto A) {
	// A: i x j

	auto A_ik = A ^ noarr::rename<'j', 'k'>();
	auto A_jk = A ^ noarr::rename<'i', 'j', 'j', 'k'>();

	noarr::traverser(A, A_ik, A_jk)
		.template for_dims<'i'>([=](auto inner) {
			auto state = inner.state();

			inner
				.order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state)))
				.template for_dims<'j'>([=](auto inner) {
					auto state = inner.state();
					
					inner
						.order(noarr::slice<'k'>(0, noarr::get_index<'j'>(state)))
						.template for_each<'k'>([=](auto state) {
							A[state] -= A_ik[state] * A_jk[state];
						});

					A[state] /= (A ^ noarr::fix<'i'>(noarr::get_index<'j'>(state)))[state];
				});

			auto A_ii = A ^ noarr::fix<'j'>(noarr::get_index<'i'>(state));

			inner
				.order(noarr::slice<'k'>(0, noarr::get_index<'i'>(state)))
				.template for_each<'k'>([=](auto state) {
					A_ii[state] -= A_ik[state] * A_ik[state];
				});

			A_ii[state] = std::sqrt(A_ii[state]);
		});
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// data
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(n, n));

	// initialize data
	init_array(A.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_cholesky(A.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argv[0] != ""s) [A = A.get_ref()] {
		std::cout << std::fixed << std::setprecision(2);
		noarr::traverser(A)
			.template for_dims<'i'>([=](auto inner) {
				auto state = inner.state();

				inner
					.order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state) + 1))
					.template for_each<'j'>([=](auto state) {
						std::cout << A[state] << " ";
					});

				std::cout << std::endl;
			});
	}();

	std::cerr << duration.count() << std::endl;
}
