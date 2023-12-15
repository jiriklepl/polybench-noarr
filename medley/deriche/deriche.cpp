#include <cmath>
#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "deriche.hpp"

using num_t = DATA_TYPE;

namespace {

// initialization function
void init_array(num_t &alpha, auto imgIn, auto) noexcept {
	// imgIn: w x h
	// imgOut: w x h

	alpha = (num_t)0.25;

	noarr::traverser(imgIn)
		.for_each([=](auto state) constexpr noexcept {
			auto [w, h] = noarr::get_indices<'w', 'h'>(state);

			imgIn[state] = (num_t)((313 * w + 991 * h) % 65536) / 65535.0f;
		});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_deriche(num_t alpha, auto imgIn, auto imgOut, auto y1, auto y2) noexcept {
	// imgIn: w x h
	// imgOut: w x h
	// y1: w x h
	// y2: w x h

	num_t k;
	num_t a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, c1, c2;
	k = ((num_t)1.0 - std::exp(-alpha)) * ((num_t)1.0 - std::exp(-alpha)) / ((num_t)1.0 + (num_t)2.0 * alpha * std::exp(-alpha) - std::exp(((num_t)2.0 * alpha)));
	a1 = a5 = k;
	a2 = a6 = k * std::exp(-alpha) * (alpha - (num_t)1.0);
	a3 = a7 = k * std::exp(-alpha) * (alpha + (num_t)1.0);
	a4 = a8 = -k * std::exp(((num_t)(-2.0) * alpha));
	b1 = std::pow((num_t)2.0, -alpha);
	b2 = -std::exp(((num_t)(-2.0) * alpha));
	c1 = c2 = 1;

	#pragma scop
	noarr::traverser(imgIn, y1)
		.template for_dims<'w'>([=](auto inner) constexpr noexcept {
			num_t ym1 = 0;
			num_t ym2 = 0;
			num_t xm1 = 0;

			inner.for_each([=, &ym1, &ym2, &xm1](auto state) constexpr noexcept {
				y1[state] = a1 * imgIn[state] + a2 * xm1 + b1 * ym1 + b2 * ym2;
				xm1 = imgIn[state];
				ym2 = ym1;
				ym1 = y1[state];
			});
		});

	noarr::traverser(imgIn, y2)
		.template for_dims<'w'>([=](auto inner) constexpr noexcept {
			num_t yp1 = 0;
			num_t yp2 = 0;
			num_t xp1 = 0;
			num_t xp2 = 0;

			inner
				.order(noarr::reverse<'h'>())
				.for_each([=, &yp1, &yp2, &xp1, &xp2](auto state) constexpr noexcept {
					y2[state] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
					xp2 = xp1;
					xp1 = imgIn[state];
					yp2 = yp1;
					yp1 = y2[state];
				});
		});


	noarr::traverser(y1, y2, imgOut)
		.for_each([=](auto state) constexpr noexcept {
			imgOut[state] = c1 * (y1[state] + y2[state]);
		});

	noarr::traverser(imgOut, y1)
		.template for_dims<'h'>([=](auto inner) constexpr noexcept {
			num_t tm1 = 0;
			num_t ym1 = 0;
			num_t ym2 = 0;

			inner.for_each([=, &tm1, &ym1, &ym2](auto state) constexpr noexcept {
				y1[state] = a5 * imgOut[state] + a6 * tm1 + b1 * ym1 + b2 * ym2;
				tm1 = imgOut[state];
				ym2 = ym1;
				ym1 = y1[state];
			});
		});

	noarr::traverser(imgOut, y2)
		.template for_dims<'h'>([=](auto inner) constexpr noexcept {
			num_t tp1 = 0;
			num_t tp2 = 0;
			num_t yp1 = 0;
			num_t yp2 = 0;

			inner
				.order(noarr::reverse<'w'>())
				.for_each([=, &tp1, &tp2, &yp1, &yp2](auto state) constexpr noexcept {
					y2[state] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
					tp2 = tp1;
					tp1 = imgOut[state];
					yp2 = yp1;
					yp1 = y2[state];
				});
		});

	noarr::traverser(y1, y2, imgOut).for_each([=](auto state) constexpr noexcept {
		imgOut[state] = c2 * (y1[state] + y2[state]);
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t nw = NW;
	std::size_t nh = NH;

	// data
	num_t alpha;
	auto imgIn = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'w', 'h'>(nw, nh));
	auto imgOut = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'w', 'h'>(nw, nh));

	auto y1 = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'w', 'h'>(nw, nh));
	auto y2 = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'w', 'h'>(nw, nh));

	// initialize data
	init_array(alpha, imgIn.get_ref(), imgOut.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_deriche(alpha, imgIn.get_ref(), imgOut.get_ref(), y1.get_ref(), y2.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, imgOut.get_ref() ^ noarr::hoist<'w'>());
	}

	std::cerr << duration.count() << std::endl;
}
