#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/planner.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "bicg.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec =  noarr::vector<'i'>();
constexpr auto j_vec =  noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto A, auto r, auto p) {
	// A: i x j
	// r: i
	// p: j
	using namespace noarr;

	auto ni = A | get_length<'i'>();
	auto nj = A | get_length<'j'>();

	traverser(p) | [=](auto state) {
		auto j = get_index<'j'>(state);
		p[state] = (num_t)(j % nj) / nj;
	};

	traverser(A, r) | for_dims<'i'>([=](auto inner) {
		auto i = get_index<'i'>(inner);

		r[inner] = (num_t)(i % ni) / ni;

		inner | [=](auto state) {
			auto j = get_index<'j'>(state);

			A[state] = (num_t)(i * (j + 1) % ni) / ni;
		};
	});
}

// computation kernel
template<class Order = noarr::neutral_proto>
[[gnu::flatten, gnu::noinline]]
void kernel_bicg(auto A, auto s, auto q, auto p, auto r, Order order = {}) {
	// A: i x j
	// s: j
	// q: i
	// p: j
	// r: i
	using namespace noarr;

	#pragma scop
	traverser(s) | [=](auto state) {
		s[state] = 0;
	};

	planner(A, s, q, p, r) ^ for_each_elem([](auto &&A, auto &&s, auto &&q, auto &&p, auto &&r) {
		s += A * r;
		q += A * p;
	}) ^ for_dims<'i'>([=](auto inner) {
		q[inner] = 0;
		inner();
	}) ^ order | planner_execute();
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nj = NJ;

	// data
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ noarr::set_length<'i'>(ni) ^ noarr::set_length<'j'>(nj));

	auto s = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'j'>(nj));
	auto q = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(ni));

	auto p = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'j'>(nj));
	auto r = noarr::make_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(ni));

	// initialize data
	init_array(A.get_ref(), r.get_ref(), p.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_bicg(A.get_ref(), s.get_ref(), q.get_ref(), p.get_ref(), r.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, s);
		noarr::serialize_data(std::cout, q);
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
