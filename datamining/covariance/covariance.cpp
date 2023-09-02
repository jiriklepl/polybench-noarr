#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/planner.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "covariance.hpp"

// autotuning
#include "test.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec =  noarr::vector<'i'>();
constexpr auto j_vec =  noarr::vector<'j'>();
constexpr auto k_vec =  noarr::vector<'k'>();

struct tuning {
	NOARR_TUNE_BEGIN(opentuner_formatter( \
		std::cout, \
		std::make_shared<noarr::tuning::cmake_compile_command_builder>("../..", "build", "covariance", "-DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -D_POSIX_C_SOURCE=200809L"), \
		std::make_shared<noarr::tuning::direct_run_command_builder>("build/covariance"), \
		"return Result(time=float(run_result['stderr'].split()[0]))"));

	NOARR_TUNE_PAR(block_i, noarr::tuning::choice,
		noarr::bcast<'I'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_k, noarr::tuning::choice,
		noarr::bcast<'K'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<64>));

	NOARR_TUNE_PAR(order, noarr::tuning::choice,
		*block_i ^ *block_k,
		*block_k ^ *block_i);

	NOARR_TUNE_PAR(data_layout, noarr::tuning::choice,
		k_vec ^ j_vec,
		j_vec ^ k_vec);

	NOARR_TUNE_PAR(cov_layout, noarr::tuning::choice,
		i_vec ^ j_vec,
		j_vec ^ i_vec);

	NOARR_TUNE_END();
} tuning;

// initialization function
void init_array(num_t &float_n, auto data) {
	// data: k x j

	float_n = data | noarr::get_length<'k'>();

	noarr::traverser(data).for_each([=](auto state) {
		auto [k, j] = noarr::get_indices<'k', 'j'>(state);
		data[state] = (num_t)(k * j) / (data | noarr::get_length<'j'>());
	});
}

// computation kernel
template<class Order = noarr::neutral_proto>
void kernel_covariance(num_t float_n, auto data, auto cov, auto mean, Order order = {}) {
	// data: k x j
	// cov: i x j
	// mean: j

	auto cov_ji = cov ^ noarr::rename<'i', 'j', 'j', 'i'>();
	auto data_ki = data ^ noarr::rename<'j', 'i'>();

	noarr::traverser(mean).for_each([=](auto state) {
		mean[state] = 0;
	});

	noarr::traverser(data, mean).for_each([=](auto state) {
		mean[state] += data[state];
	});

	noarr::traverser(mean).for_each([=](auto state) {
		mean[state] /= float_n;
	});

	noarr::traverser(data, mean).for_each([=](auto state) {
		data[state] -= mean[state];
	});

	noarr::traverser(cov).template for_dims<'i'>([=](auto inner) {
		inner
			.order(noarr::shift<'j'>(noarr::get_index<'i'>(inner.state())))
			.for_each([=](auto state) {
				cov[state] = 0;
			});
	});

	noarr::planner(data, cov, mean)
		.for_each([=](auto state) {
			cov[state] += data[state] * data_ki[state];
		})
		.template for_sections<'i'>([=](auto inner) {
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
		.template for_dims<'i'>([=](auto inner) {
			inner
				.order(noarr::shift<'j'>(noarr::get_index<'i'>(inner.state())))
				.for_each([=](auto state) {
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
	auto data = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.data_layout ^ set_lengths);
	auto cov = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.cov_layout ^ set_lengths);
	auto mean = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'j'>(nj));

	// initialize data
	init_array(float_n, data.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_covariance(float_n, data.get_ref(), cov.get_ref(), mean.get_ref(), *tuning.order);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, cov.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << duration.count() << std::endl;
}
