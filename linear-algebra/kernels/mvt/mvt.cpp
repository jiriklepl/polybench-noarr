#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "mvt.hpp"

// autotuning
#include <noarr/structures/tuning/formatters/opentuner_formatter.hpp>

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec =  noarr::vector<'i'>();
constexpr auto j_vec =  noarr::vector<'j'>();

struct tuning {
	NOARR_TUNE_BEGIN(noarr::tuning::opentuner_formatter(
		std::cout,
		std::make_shared<noarr::tuning::cmake_compile_command_builder>("../..", "build", "mvt", "-D" STRINGIFY(DATASET_SIZE) " -D" STRINGIFY(DATA_TYPE_CHOICE)),
		std::make_shared<noarr::tuning::direct_run_command_builder>("build/mvt"),
		"return Result(time=float(run_result['stderr'].split()[0]))"));

	NOARR_TUNE_PAR(block_i1, noarr::tuning::choice,
		noarr::bcast<'I'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_j1, noarr::tuning::choice,
		noarr::bcast<'J'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_i2, noarr::tuning::choice,
		noarr::bcast<'I'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'i', 'I', 'i', 's'>(noarr::lit<64>));

	NOARR_TUNE_PAR(block_j2, noarr::tuning::choice,
		noarr::bcast<'J'>(noarr::lit<1>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<2>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<4>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<8>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<16>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<32>),
		noarr::strip_mine_dynamic<'j', 'J', 'j', 't'>(noarr::lit<64>));

	NOARR_TUNE_PAR(order1, noarr::tuning::choice,
		*block_i1 ^ *block_j1,
		*block_j1 ^ *block_i1);

	NOARR_TUNE_PAR(order2, noarr::tuning::choice,
		*block_i2 ^ *block_j2,
		*block_j2 ^ *block_i2);

	NOARR_TUNE_PAR(a_layout, noarr::tuning::choice,
		i_vec ^ j_vec);

	NOARR_TUNE_END();
} tuning;

// initialization function
void init_array(auto x1, auto x2, auto y1, auto y2, auto A) noexcept {
	// x1: i
	// x2: i
	// y1: j
	// y2: j
	// A: i x j

	auto n = A | noarr::get_length<'i'>();

	auto y1_i = y1 ^ noarr::rename<'j', 'i'>();
	auto y2_i = y2 ^ noarr::rename<'j', 'i'>();

	noarr::traverser(x1, x2, y1_i, y2_i, A)
		.template for_dims<'i'>([=](auto inner) {
			auto state = inner.state();
			auto i = noarr::get_index<'i'>(state);
			
			x1[state] = (num_t)(i % n) / n;
			x2[state] = (num_t)((i + 1) % n) / n;
			y1_i[state] = (num_t)((i + 3) % n) / n;
			y2_i[state] = (num_t)((i + 4) % n) / n;

			inner.for_each([=](auto state) {
				auto j = noarr::get_index<'j'>(state);

				A[state] = (num_t)(i * j % n) / n;
			});
		});
}

// computation kernel
template<class Order1 = noarr::neutral_proto, class Order2 = noarr::neutral_proto>
[[gnu::flatten, gnu::noinline]]
void kernel_mvt(auto x1, auto x2, auto y1, auto y2, auto A, Order1 order1 = {}, Order2 order2 = {}) noexcept {
	// x1: i
	// x2: i
	// y1: j
	// y2: j
	// A: i x j

	auto A_ji = A ^ noarr::rename<'i', 'j', 'j', 'i'>();

	noarr::traverser(x1, A, y1)
		.order(order1)
		.for_each([=](auto state) {
			x1[state] += A[state] * y1[state];
		});

	noarr::traverser(x2, A_ji, y2)
		.order(order2)
		.for_each([=](auto state) {
			x2[state] += A_ji[state] * y2[state];
		});
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// data
	auto x1 = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));
	auto x2 = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));

	auto y1 = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'j'>(n));
	auto y2 = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'j'>(n));

	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ *tuning.a_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));

	// initialize data
	init_array(x1.get_ref(), x2.get_ref(), y1.get_ref(), y2.get_ref(), A.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_mvt(x1.get_ref(), x2.get_ref(), y1.get_ref(), y2.get_ref(), A.get_ref(), *tuning.order1, *tuning.order2);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, x1);
		noarr::serialize_data(std::cout, x2);
	}

	std::cerr << duration.count() << std::endl;
}
