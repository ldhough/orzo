#include <iostream>
#include <string>
#include <cstring>
#include <set>
#include <pasta/bit_vector/bit_vector.hpp>
#include <pasta/bit_vector/support/rank_select.hpp>
#include <pasta/bit_vector/support/rank.hpp>
#include <pasta/bit_vector/support/flat_rank.hpp>
#include <pasta/bit_vector/support/flat_rank_select.hpp>
#include <orzo/orzo.h>
#include <orzo/utils.h>
#include <orzo/bitvector.h>

#ifdef __linux__
#include <sched.h>
#endif

using std::cerr, std::endl, std::cout;

volatile char *cache_bytes = nullptr;
volatile size_t cache_sum = 0;

void flush_cache(size_t nbytes = 1048576) {
    cache_bytes = nullptr;
    cache_sum = 0;
    char *bytes = (char*) malloc(nbytes);
    memset(bytes, 1, nbytes);
    cache_bytes = bytes;
    size_t sum = 0;
    for (size_t idx = 0; idx < nbytes * 10; idx++) {
        sum += (size_t) bytes[random_integer<size_t>(0, nbytes)];
    }
    cache_sum = sum;
}

void set_affinity(size_t id = 1) {
#ifdef __linux__
    cerr << "setting affinity" << endl;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(id, &mask);
    int status = sched_setaffinity(0, sizeof(mask), &mask);
    if (status != 0) cerr << "failed to set affinity" << endl;
#endif
    return;
}

void compare(std::string query_type, size_t size, size_t sparsity, size_t seed) {
    set_affinity();
    bool do_rank = query_type == "rank";
    if (do_rank) {
        cerr << "Query type: rank" << endl;
    } else {
        cerr << "Query type: select" << endl;
    }
    cerr << "Seed is: " << seed << endl;
    cerr << "BV size is: " << size << endl;
    cerr << "BV sparsity is: " << sparsity << endl;
    uint64_t query_count = 10000000;
    std::vector<uint64_t> orzo_rank_v;
    std::vector<uint64_t> orzo_select_v;
    std::vector<uint64_t> pasta_rank_v;
    std::vector<uint64_t> pasta_select_v;
    pasta::BitVector bv(size, 0);
    OrzoBitvector pssg_bv(size, 5632);
    size_t hot_count = 0;
    for (size_t i = 0; i < bv.size(); i++) {
        auto x = random_integer<size_t>(1, 100, seed);
        if (x > sparsity) {
            //cerr << "wut?" << endl;
            bv[i] = 1;
            pssg_bv.set_bit(i);
            ++hot_count;
        }
    }
    uint64_t *bv2 = pssg_bv.data();
    std::vector<size_t> access_order;
    cerr << "Hot bits: " << hot_count << endl;
    for (size_t idx = 0; idx < query_count; idx++) {
        if (do_rank) {
            access_order.push_back(random_integer<size_t>(1, size));
        } else { // for select
            size_t rand = random_integer<size_t>(1, hot_count, seed);
            access_order.push_back(rand);
        }
    }
    cerr << "Running benchmarks..." << endl;
    pasta::RankSelect poppy(bv);
    pasta::FlatRankSelect pasta_flat(bv);
    Orzo orzo(bv2, size);
    if (do_rank) {
        // POPPY RANK
        flush_cache();
        [[maybe_unused]]
        static volatile size_t i1 = 0;
        auto poppy_rank_start = std::chrono::system_clock::now();
        for (size_t idx = 0; idx < query_count; idx++) {
            [[maybe_unused]]
            size_t unused = poppy.rank1(access_order[idx]);
            i1 = unused;
        }
        auto poppy_rank_end = std::chrono::system_clock::now();
        std::chrono::duration<double> poppy_rank_elapsed = poppy_rank_end - poppy_rank_start;
        cerr << "finished poppy rank" << endl;
        // ORZO RANK
        flush_cache();
        [[maybe_unused]]
        static volatile size_t i3 = 0;
        auto orzo_rank_start = std::chrono::system_clock::now();
        for (size_t idx = 0; idx < query_count; idx++) {
            [[maybe_unused]]
            size_t unused = orzo.rank1(bv2, access_order[idx]);
            i3 = unused;
#ifdef CHECK_CORRECTNESS
            orzo_rank_v.push_back(unused);
#endif
        }
        auto orzo_rank_end = std::chrono::system_clock::now();
        std::chrono::duration<double> orzo_rank_elapsed = orzo_rank_end - orzo_rank_start;
        cerr << "finished orzo rank" << endl;
        // PASTA RANK 
        flush_cache();
        [[maybe_unused]]
        static volatile size_t i5 = 0;
        auto pasta_rank_start = std::chrono::system_clock::now();
        for (size_t idx = 0; idx < query_count; idx++) {
            [[maybe_unused]]
            size_t unused = pasta_flat.rank1(access_order[idx]);
            i5 = unused;
#ifdef CHECK_CORRECTNESS
            pasta_rank_v.push_back(unused);
#endif
        }
        auto pasta_rank_end = std::chrono::system_clock::now();
        std::chrono::duration<double> pasta_rank_elapsed = pasta_rank_end - pasta_rank_start;
        cerr << "finished pasta rank" << endl;

        orzo_rank_elapsed /= query_count;
        pasta_rank_elapsed /= query_count;
        poppy_rank_elapsed /= query_count;
        cerr << "Elapsed time for pasta_flat rank: " << pasta_rank_elapsed.count() << endl;
        cerr << "Elapsed time for poppy rank: " << poppy_rank_elapsed.count() << endl;
        cerr << "Elapsed time for orzo rank: " << orzo_rank_elapsed.count() << endl;
        cout << "pasta," << query_type << "," << sparsity
            << "," << size << "," << pasta_rank_elapsed.count() << endl;
        cout << "poppy," << query_type << "," << sparsity
            << "," << size << "," << poppy_rank_elapsed.count() << endl;
        cout << "orzo," << query_type << "," << sparsity
            << "," << size << "," << orzo_rank_elapsed.count() << endl;
#ifdef CHECK_CORRECTNESS
        bool correct_orzo = true;
        size_t incorrect_count_orzo = 0;
        for (size_t i = 0; i < pasta_rank_v.size(); i++) {
            if (pasta_rank_v[i] != orzo_rank_v[i]) {
                correct_orzo = false;
                incorrect_count_orzo++;
                if (incorrect_count_orzo < 10) {
                    cerr << "incorrect orzo rank index: " << i << endl;
                }
            }
        }
        cerr << ((correct_orzo) ? "correct_orzo_rank" : "incorrect_orzo_rank") << endl;
        cerr << "incorrect orzo rank count: " << incorrect_count_orzo << endl;
#endif
    } else {
        // POPPY SELECT -----
        flush_cache();
        [[maybe_unused]]
        static volatile size_t i2 = 0;
        auto poppy_select_start = std::chrono::system_clock::now();
        for (size_t idx = 0; idx < query_count; idx++) {
            [[maybe_unused]]
            size_t unused = poppy.select1(access_order[idx]);
            i2 = unused;
        }
        auto poppy_select_end = std::chrono::system_clock::now();
        std::chrono::duration<double> poppy_select_elapsed = poppy_select_end - poppy_select_start;
        cerr << "finished poppy select" << endl;
        // PASTA SELECT -----
        flush_cache();
        [[maybe_unused]]
        static volatile size_t i6 = 0;
        auto pasta_select_start = std::chrono::system_clock::now();
        for (size_t idx = 0; idx < query_count; idx++) {
            [[maybe_unused]]
            size_t unused = pasta_flat.select1(access_order[idx]);
            i6 = unused;
#ifdef CHECK_CORRECTNESS
            pasta_select_v.push_back(unused);
#endif
        }
        auto pasta_select_end = std::chrono::system_clock::now();
        std::chrono::duration<double> pasta_select_elapsed = pasta_select_end - pasta_select_start;
        cerr << "finished pasta select" << endl;
        // ORZO SELECT -----
        flush_cache();
        [[maybe_unused]]
        static volatile size_t i4 = 0;
        auto orzo_select_start = std::chrono::system_clock::now();
        for (size_t idx = 0; idx < query_count; idx++) {
            [[maybe_unused]]
            size_t unused = orzo.select1(bv2, access_order[idx]);
            i4 = unused;
#ifdef CHECK_CORRECTNESS
            orzo_select_v.push_back(unused);
#endif
        }
        auto orzo_select_end = std::chrono::system_clock::now();
        std::chrono::duration<double> orzo_select_elapsed = orzo_select_end - orzo_select_start;
        cerr << "finished orzo select" << endl;
        orzo_select_elapsed /= query_count;
        pasta_select_elapsed /= query_count;
        poppy_select_elapsed /= query_count;
        cerr << "Elapsed time for pasta_flat select: " << pasta_select_elapsed << endl;
        cerr << "Elapsed time for poppy select: " << poppy_select_elapsed << endl;
        cerr << "Elapsed time for orzo select: " << orzo_select_elapsed << endl;
        cout << "pasta," << query_type << "," << sparsity
            << "," << size << "," << pasta_select_elapsed.count() << endl;
        cout << "poppy," << query_type << "," << sparsity
            << "," << size << "," << poppy_select_elapsed.count() << endl;
        cout << "orzo," << query_type << "," << sparsity
            << "," << size << "," << orzo_select_elapsed.count() << endl;
#ifdef CHECK_CORRECTNESS
        bool correct_orzo_select = true;
        size_t incorrect_count_orzo_select = 0;
        //orzo_select_v[10] = 11;
        for (size_t i = 0; i < pasta_select_v.size(); i++) {
            if (pasta_select_v[i] != orzo_select_v[i]) {
                correct_orzo_select = false;
                incorrect_count_orzo_select++;
                if (incorrect_count_orzo_select < 10) {
                    cerr << "incorrect orzo select index" << i << endl;
                }
            }
        }
        cerr << ((correct_orzo_select) ? "correct_orzo_select" : "incorrect_orzo_select") << endl;
        cerr << "incorrect orzo select count: " << incorrect_count_orzo_select << endl;
#endif
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        cerr << "Usage: orzo-benchmark <query type: 'rank' or 'select'> <size of bit vector> "
            "<~bv sparsity 0-99> <rng seed>" << endl;
        return -1;
    }
    std::string query_type(argv[1]);
    size_t size = atoll(argv[2]);
    size_t sparsity = atoi(argv[3]);
    size_t seed = atoi(argv[4]);
    compare(query_type, size, sparsity, seed);
    return 0;
}
