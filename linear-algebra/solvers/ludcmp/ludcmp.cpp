#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "ludcmp.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec =  noarr::vector<'i'>();
constexpr auto j_vec =  noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto A, auto b, auto x, auto y) noexcept {
	// A: i x j
	// b: i
	// x: i
	// y: i

	int n = A | noarr::get_length<'i'>();
	num_t fn = (num_t)n;

	noarr::traverser(b, x, y)
		.for_each([=](auto state) {
			auto i = noarr::get_index<'i'>(state);

			x[state] = 0;
			y[state] = 0;
			b[state] = (i + 1) / fn / 2.0 + 4;
		});

	noarr::traverser(A)
		.template for_dims<'i'>([=](auto inner) {
			auto state = inner.state();

			auto i = noarr::get_index<'i'>(state);

			inner
				.order(noarr::slice<'j'>(0, i + 1))
				.for_each([=](auto state) {
					int j = noarr::get_index<'j'>(state);

					A[state] = (num_t)(-j % n) / n + 1;
				});

			inner
				.order(noarr::shift<'j'>(i + 1))
				.for_each([=](auto state) {
					A[state] = 0;
				});

			A[state & noarr::idx<'j'>(i)] = 1;
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

	noarr::traverser(A, B_ref).for_each([=](auto state) {
		A[state] = B_ref[state];
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_ludcmp(auto A, auto b, auto x, auto y) noexcept {
	// A: i x j
	// b: i
	// x: i
	// y: i

	auto A_ik = A ^ noarr::rename<'j', 'k'>();
	auto A_kj = A ^ noarr::rename<'i', 'k'>();

	#pragma scop
	noarr::traverser(A, b, x, y, A_ik, A_kj)
		.template for_dims<'i'>([=](auto inner) {
			auto state = inner.state();

			inner
				.order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state)))
				.template for_dims<'j'>([=](auto inner) {
					auto state = inner.state();

					num_t w = A[state];

					inner
						.order(noarr::slice<'k'>(0, noarr::get_index<'j'>(state)))
						.template for_each<'k'>([=, &w](auto state) {
							w -= A_ik[state] * A_kj[state];
						});

					A[state] = w / (A ^ noarr::fix<'i'>(noarr::get_index<'j'>(state)))[state];
				});

			inner
				.order(noarr::shift<'j'>(noarr::get_index<'i'>(state)))
				.template for_dims<'j'>([=](auto inner) {
					auto state = inner.state();

					num_t w = A[state];

					inner
						.order(noarr::slice<'k'>(0, noarr::get_index<'i'>(state)))
						.template for_each<'k'>([=, &w](auto state) {
							w -= A_ik[state] * A_kj[state];
						});

					A[state] = w;
				});
		});

		noarr::traverser(A, b, y)
			.template for_dims<'i'>([=](auto inner) {
				auto state = inner.state();

				num_t w = b[state];

				inner
					.order(noarr::slice<'j'>(0, noarr::get_index<'i'>(state)))
					.template for_each<'j'>([=, &w](auto state) {
						w -= A[state] * y[noarr::idx<'i'>(noarr::get_index<'j'>(state))];
					});

				y[state] = w;
			});

		noarr::traverser(A, x)
			.order(noarr::reverse<'i'>())
			.template for_dims<'i'>([=](auto inner) {
				auto state = inner.state();

				num_t w = y[state];

				inner
					.order(noarr::shift<'j'>(noarr::get_index<'i'>(state) + 1))
					.template for_each<'j'>([=, &w](auto state) {
						w -= A[state] * x[noarr::idx<'i'>(noarr::get_index<'j'>(state))];
					});

				x[state] = w / A[state & noarr::idx<'j'>(noarr::get_index<'i'>(state))];
			});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// data
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));
	auto b = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));
	auto x = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));
	auto y = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));

	// initialize data
	init_array(A.get_ref(), b.get_ref(), x.get_ref(), y.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_ludcmp(A.get_ref(), b.get_ref(), x.get_ref(), y.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, x);
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
