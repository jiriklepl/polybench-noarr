#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/planner.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "covariance.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec =  noarr::vector<'i'>();
constexpr auto j_vec =  noarr::vector<'j'>();
constexpr auto k_vec =  noarr::vector<'k'>();

struct tuning {
	DEFINE_PROTO_STRUCT(block_i, noarr::neutral_proto());
	DEFINE_PROTO_STRUCT(block_k, noarr::neutral_proto());

	DEFINE_PROTO_STRUCT(order, block_i ^ block_k);

	DEFINE_PROTO_STRUCT(data_layout, k_vec ^ j_vec);
	DEFINE_PROTO_STRUCT(cov_layout, i_vec ^ j_vec);
} tuning;

// initialization function
void init_array(num_t &float_n, auto data) noexcept {
	// data: k x j

	float_n = data | noarr::get_length<'k'>();

	noarr::traverser(data).for_each([=](auto state) constexpr noexcept {
		auto [k, j] = noarr::get_indices<'k', 'j'>(state);
		data[state] = (num_t)(k * j) / (data | noarr::get_length<'j'>());
	});
}

// computation kernel
template<class Order = noarr::neutral_proto>
[[gnu::flatten, gnu::noinline]]
void kernel_covariance(num_t float_n, auto data, auto cov, auto mean, Order order = {}) noexcept {
	// data: k x j
	// cov: i x j
	// mean: j

	auto cov_ji = cov ^ noarr::rename<'i', 'j', 'j', 'i'>();
	auto data_ki = data ^ noarr::rename<'j', 'i'>();

	noarr::traverser(mean).for_each([=](auto state) constexpr noexcept {
		mean[state] = 0;
	});

	noarr::traverser(data, mean).for_each([=](auto state) constexpr noexcept {
		mean[state] += data[state];
	});

	noarr::traverser(mean).for_each([=](auto state) constexpr noexcept {
		mean[state] /= float_n;
	});

	noarr::traverser(data, mean).for_each([=](auto state) constexpr noexcept {
		data[state] -= mean[state];
	});

	noarr::traverser(cov).template for_dims<'i'>([=](auto inner) constexpr noexcept {
		inner
			.order(noarr::shift<'j'>(noarr::get_index<'i'>(inner.state())))
			.for_each([=](auto state) constexpr noexcept {
				cov[state] = 0;
			});
	});

	noarr::planner(data, cov, mean)
		.for_each([=](auto state) constexpr noexcept {
			cov[state] += data[state] * data_ki[state];
		})
		.template for_sections<'i'>([](auto inner) constexpr noexcept {
			inner
				.order(noarr::shift<'j'>(noarr::get_index<'i'>(inner.state())))
				();
		})
		.order(noarr::hoist<'k'>())
		.order(noarr::hoist<'j'>())
		.order(noarr::hoist<'i'>())
		.order(order)
		();

	noarr::traverser(cov, cov_ji)
		.template for_dims<'i'>([=](auto inner) constexpr noexcept {
			inner
				.order(noarr::shift<'j'>(noarr::get_index<'i'>(inner.state())))
				.for_each([=](auto state) constexpr noexcept {
					cov[state] /= float_n - (num_t)1;
					cov_ji[state] = cov[state];
				});
		});
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t nk = NK;
	std::size_t nj = NJ;

	auto set_lengths = noarr::set_length<'k'>(nk) ^ noarr::set_length<'j'>(nj) ^ noarr::set_length<'i'>(nj);

	// data
	num_t float_n;
	auto data = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.data_layout ^ set_lengths);
	auto cov = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.cov_layout ^ set_lengths);
	auto mean = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'j'>(nj));

	// initialize data
	init_array(float_n, data.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_covariance(float_n, data.get_ref(), cov.get_ref(), mean.get_ref(), tuning.order);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, cov.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << duration.count() << std::endl;
}
