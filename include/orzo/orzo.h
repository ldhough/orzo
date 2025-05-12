#ifndef ORZO_H
#define ORZO_H

#include <cstdint>
#include <cstdlib>
#include <climits>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <bit>
#include <bitset>
#include <vector>
#include <iostream>
#include <immintrin.h>
#include "utils.h"

using std::cout, std::endl;

template<
    uint64_t BASIC_BLOCK_COUNT = 512,
    uint64_t L1L2_COUNT = 128,
    uint64_t N_L2 = 10,
    bool use_l0 = true,
    bool support_select = true
>
class Orzo {

    private:

        uint64_t bv_count;
        uint64_t one_count;
        uint64_t *l0;
        /*
         * orzo reduces the width of the L1 indices to 18 bits, and so this
         * is the size of an upper block. This turns out not to be a major
         * performance problem for rank queries, but select queries need to
         * scan or binary search the l0 index, which is more challenging
         * with more entries. Hence, we use a different upper block size for
         * rank and for select, and use ~2**32 bit upper blocks for select).
         */
        uint64_t *select_l0;
        uint32_t **select_samples;
        //uint32_t **select_sample_ptrs;
        __uint128_t *l1l2; // interleaved l1 and l2 indices

        // counts are of bits, sizes are in bytes
        static constexpr uint64_t BASIC_BLOCK_WORDS = BASIC_BLOCK_COUNT / 64;
        static constexpr uint64_t LOWER_BLOCK_COUNT = (N_L2 + 1) * 512; // 5632
        static constexpr uint64_t LOWER_BLOCK_WORDS = LOWER_BLOCK_COUNT / 64;
        static constexpr uint64_t L2_UNIVERSE = N_L2 * 512;
        static constexpr uint64_t EF_UPPER_BV_COUNT = 2 * N_L2;
        static constexpr uint64_t EF_UPPER_ELE_COUNT = 2;
        const uint64_t EF_LOWER_BV_COUNT = N_L2 * (uint64_t) std::ceil(std::log2((float) L2_UNIVERSE / (float) N_L2));
        const uint64_t EF_LOWER_ELE_COUNT = EF_LOWER_BV_COUNT / N_L2;
        const uint64_t EF_TOTAL_COUNT = EF_UPPER_BV_COUNT + EF_LOWER_BV_COUNT;
        // pow(2, L1L2_COUNT - EF_TOTAL_COUNT)
        // temporarily hardcoded for L1L2_COUNT = 10, needs to be evenly divisible by lower block count
        const uint64_t UPPER_BLOCK_COUNT = 259072; //2ul << ((L1L2_COUNT - EF_TOTAL_COUNT) - 1);
        const uint64_t LOWER_PER_UPPER = UPPER_BLOCK_COUNT / LOWER_BLOCK_COUNT;
        static constexpr uint64_t BASIC_BLOCK_SIZE = (BASIC_BLOCK_COUNT / 8);
        static constexpr uint64_t LOWER_BLOCK_SIZE = (LOWER_BLOCK_COUNT / 8);
        const uint64_t UPPER_BLOCK_SIZE = (UPPER_BLOCK_COUNT / 8);
        // ceil(log2(N_L2)) can't be constexpr in C++20, 23 so this should work the same for integer N_L2
        static constexpr uint64_t EF_UPPER_SPLIT_COUNT = (63 - std::countl_zero(N_L2)) + ((uint64_t) !std::has_single_bit(N_L2));
        // number of buckets necessary is the upper bits of max
        // number represented, +1 to account for the zero bucket,
        // can't just shift by EF_LOWER_ELE_COUNT bc not constexpr
        static constexpr uint64_t NUM_BUCKETS = (L2_UNIVERSE >> (64 - (std::countl_zero(L2_UNIVERSE) + EF_UPPER_SPLIT_COUNT))) + 1;
        const uint64_t EF_LOWER_MASK = (2 << (EF_LOWER_ELE_COUNT - 1)) - 1;
        const uint64_t EF_UPPER_BV_MASK = (2 << (EF_UPPER_BV_COUNT - 1)) - 1; 
        static constexpr uint64_t SELECT_SAMPLE = 8192; //11264; //8192;
        // (2 ** 32) - 4096 so that it is evenly divisible by 5632 AND by UPPER_BLOCK_COUNT, simplifies select logic
        static constexpr uint64_t SELECT_UPPER_BLOCK_COUNT = 4294895616; //4294963200; //4294967296; // 2 ** 32
        static constexpr uint64_t L1L2_PER_SELECT_UPPER = SELECT_UPPER_BLOCK_COUNT / LOWER_BLOCK_COUNT;

        uint64_t SELECT_L0_ENTRY_COUNT;
        uint64_t L1L2_INDEX_COUNT;

    public:

        // [ l1 | ef_upper: end ... start | ef_lower: nth ... 0th ]
        __uint128_t elias_fano_encode(uint64_t *elements) {
            __uint128_t result = 0;
            uint64_t buckets[NUM_BUCKETS] = {0};
            for (uint64_t i = 0; i < N_L2; ++i) {
                uint64_t element = elements[i];
                uint64_t upper = element >> EF_LOWER_ELE_COUNT;
                ++(buckets[upper]);
            }
            uint64_t counter = 0;
            for (uint64_t i = 0; i < NUM_BUCKETS; ++i) {
                uint64_t bucket_count = buckets[i];
                for (uint64_t j = 0; j < bucket_count; ++j) {
                    result |= (1 << counter);
                    counter++;
                }
                counter++;
            }
            for (uint64_t i = 0; i < N_L2; ++i) {
                // mask is EF_LOWER_ELE_COUNT bits all 1
                __uint128_t lower = elements[i] & EF_LOWER_MASK;
                uint64_t shift = ((i * EF_LOWER_ELE_COUNT) + EF_UPPER_BV_COUNT);
                lower <<= shift;
                result |= lower;
            }
            return result;
        }

        uint64_t *get_l0() { return this->l0; }
        __uint128_t *get_l1l2() { return this->l1l2; }
        uint64_t get_one_count() { return this->one_count; }


        Orzo(uint64_t *bv, size_t bv_count) : bv_count(bv_count) {
            size_t l0_count = (size_t) std::ceil((float) bv_count / (float) UPPER_BLOCK_COUNT);
            size_t num_lower_blocks = (size_t) std::ceil((float) bv_count / (float) LOWER_BLOCK_COUNT);
            size_t num_basic_blocks = (size_t) std::ceil((float) bv_count / (float) BASIC_BLOCK_COUNT);
            size_t bb_per_lower = LOWER_BLOCK_COUNT / BASIC_BLOCK_COUNT;
            size_t bb_per_upper = UPPER_BLOCK_COUNT / BASIC_BLOCK_COUNT;
            this->l0 = new uint64_t[l0_count + 1]();
            this->l1l2 = new __uint128_t[num_lower_blocks]();
            this->L1L2_INDEX_COUNT = num_lower_blocks;
            size_t select_l0_count = (size_t) std::ceil((float) bv_count / (float) SELECT_UPPER_BLOCK_COUNT);
            this->SELECT_L0_ENTRY_COUNT = select_l0_count;
            size_t bb_per_select_upper = SELECT_UPPER_BLOCK_COUNT / BASIC_BLOCK_COUNT;
            if constexpr(support_select) {
                this->select_l0 = new uint64_t[select_l0_count + 1]();
            }
            size_t l0_idx = 1;
            size_t select_l0_idx = 1;
            size_t l1l2_idx = 0;
            size_t hot_count_total = 0;
            size_t count_within_upper = 0;
            //size_t count_within_select_upper = 0;
            size_t count_within_lower = 0;
            // first entry (index 0) is never used, exists to simplify logic, the first
            // l2 entry is at index 1, second at index 2, etc.
            uint64_t l2_counts[N_L2 + 1] = {0};
            // increment over all basic blocks
            for (size_t i = 1; i <= num_basic_blocks; ++i) {
                size_t bb_offset = (i - 1) * (BASIC_BLOCK_SIZE / 8);
                uint64_t *bb_start = &(bv[bb_offset]);
                size_t num_words = BASIC_BLOCK_SIZE / sizeof(*bv);
                uint64_t basic_count = 0;
                for (size_t j = 0; j < num_words; ++j) {
                    basic_count += (uint64_t) std::popcount(bb_start[j]);
                }
                hot_count_total += basic_count;
                count_within_lower += basic_count;
                l2_counts[i % (N_L2 + 1)] = count_within_lower;
                bool end_of_lower = not ((bool) (i % bb_per_lower));
                if (end_of_lower) {
                    this->l1l2[l1l2_idx] |= ((__uint128_t) count_within_upper << EF_TOTAL_COUNT);
                    __uint128_t elias_fano_l2s = this->elias_fano_encode(l2_counts + 1);
                    this->l1l2[l1l2_idx] |= elias_fano_l2s;
                    count_within_upper += count_within_lower;
                    //count_within_select_upper += count_within_lower;
                    count_within_lower = 0;
                    ++l1l2_idx;
                }
                bool end_of_upper = not ((bool) (i % bb_per_upper));
                if (end_of_upper) {
                    this->l0[l0_idx] = hot_count_total;
                    count_within_upper = 0;
                    ++l0_idx;
                }
                if constexpr(support_select) {
                    bool end_of_select_upper = not ((bool) (i % bb_per_select_upper));
                    if (end_of_select_upper) {
                        this->select_l0[select_l0_idx] = hot_count_total;
                        //count_within_select_upper = 0;
                        ++select_l0_idx;
                    }
                }
            }
            this->l0[l0_idx] = hot_count_total;
            this->l1l2[l1l2_idx] |= ((__uint128_t) count_within_upper << EF_TOTAL_COUNT);
            __uint128_t elias_fano_l2s = this->elias_fano_encode(l2_counts + 1);
            this->l1l2[l1l2_idx] |= elias_fano_l2s;
            this->one_count = hot_count_total;
            // using a vector here for convenience but the data is copied to
            // the select_samples allocation with a fixed size to ensure std::vector
            // doesn't use more memory behind the scenes than is necessary
            std::vector<std::vector<uint32_t>> select_samples_tmp;
            for (size_t i = 0; i < select_l0_count; ++i) {
                std::vector<uint32_t> tmp;
                select_samples_tmp.push_back(tmp);
            }
            if constexpr(support_select) {
                this->select_l0[select_l0_idx] = hot_count_total;
                this->select_samples = (uint32_t**) calloc(select_l0_count, sizeof(uint32_t*));
                for (size_t i = 0; i < select_l0_count; ++i) {
//                    select_samples_tmp[i].push_back(0);
                    size_t cum = 0;
                    size_t next = 1;
                    size_t words_per_sel_upper = SELECT_UPPER_BLOCK_COUNT / 64;
                    size_t words_in_bucket = words_per_sel_upper;
                    if (i == (select_l0_count - 1)) {
                        size_t partial = (bv_count / 64) % words_in_bucket;
                        if (partial) {
                            words_in_bucket = partial;
                        }
                    }
                    for (size_t j = 0; j < words_in_bucket; ++j) {
                        size_t full_word_idx = (i * words_per_sel_upper) + j;
                        size_t popc = std::popcount(bv[full_word_idx]);
                        cum += popc;
                        if (cum >= next) {
                            size_t local_l1l2_idx = j / LOWER_BLOCK_WORDS;
                            select_samples_tmp[i].push_back(local_l1l2_idx);
                            next += SELECT_SAMPLE;
                        }
                    }
                }
                for (size_t i = 0; i < select_samples_tmp.size(); ++i) {
                    uint64_t bucket_size = select_samples_tmp[i].size();
                    if (bucket_size == 0) { // at least one
                        select_samples_tmp[i].push_back(0);
                        bucket_size++;
                    }
                    this->select_samples[i] = (uint32_t*) calloc(bucket_size, sizeof(uint32_t));
                    for (size_t j = 0; j < bucket_size; ++j) {
                        this->select_samples[i][j] = select_samples_tmp[i][j];
                    }
                }
            }
        }

        ~Orzo() {
            delete[] l0;
            delete[] l1l2;
            delete[] select_l0;
            for (size_t i = 0; i < this->SELECT_L0_ENTRY_COUNT; ++i) {
                free(this->select_samples[i]);
            }
            free(this->select_samples);
        }

        // assumes a bit layout like so:
        // | 63 ... 1 0 | 127 ... 65 64 |
        uint64_t rank1(uint64_t *bv, uint64_t i) {
            uint64_t l1l2_idx = i / LOWER_BLOCK_COUNT;
            __uint128_t l1l2 = this->l1l2[l1l2_idx];
            uint64_t l1_count = (uint64_t) (l1l2 >> EF_TOTAL_COUNT);
            uint64_t rank = l1_count;
            if constexpr(use_l0) {
                uint64_t l0_count = this->l0[i / UPPER_BLOCK_COUNT];
                rank += l0_count;
            }
            // distance into lower block in bits, [0, LOWER_BLOCK_COUNT)
            uint64_t j = i - (l1l2_idx * LOWER_BLOCK_COUNT);
            // idx of basic block within lower block that i (and j) present in
            uint64_t iob = (j / BASIC_BLOCK_COUNT);
            if (iob) { // if 0 only do popcnts otherwise EF decode l2
                uint64_t iob_dec = iob - 1;
                uint64_t ef_lower = EF_LOWER_MASK & (l1l2 >> (EF_UPPER_BV_COUNT + (iob_dec * EF_LOWER_ELE_COUNT)));
                // select1(iob)
                uint64_t select_result = _tzcnt_u64(_pdep_u64(1ul << iob_dec, l1l2)) + 1;
                uint64_t ef_upper = select_result - iob;
                rank += ef_lower | (ef_upper << EF_LOWER_ELE_COUNT);
            }
            uint64_t bb_offset = (i / BASIC_BLOCK_COUNT) * 8;
            uint64_t bits_considered = (i % BASIC_BLOCK_COUNT);
            uint64_t num_popcounts = bits_considered / 64;
            // full popcounts
            uint64_t ii = 0;
            for (; ii < num_popcounts; ++ii) {
                uint64_t word = bv[bb_offset + ii];
                rank += (uint64_t) std::popcount(word);
            }
            // partial popcount
            bits_considered %= 64;
            if (bits_considered) {
                uint64_t word = bv[bb_offset + ii];
                uint64_t shift = 64 - bits_considered;
                word <<= shift;
                rank += (uint64_t) std::popcount(word);
            }
            return rank;
        }

        uint64_t rank0(uint64_t *bv, uint64_t i) {
            return 1 + (i - rank1(bv, i));
        }
        
        uint64_t select1(uint64_t *bv, uint64_t i) {
            uint64_t l0_idx = 0;
            while (((l0_idx + 1) < this->SELECT_L0_ENTRY_COUNT) && (this->select_l0[l0_idx + 1] < i)) {
                ++l0_idx;
            }
            // now this is just the rank we want *within* an upper select block
            uint64_t l0 = this->select_l0[l0_idx];
            uint64_t rank = i - l0;
            uint32_t *sample_bucket = select_samples[l0_idx];
            // this idx is *within* an upper select block
            uint64_t l1l2_idx = sample_bucket[(rank - 1) / SELECT_SAMPLE];
            // make it a full l1l2_idx
            l1l2_idx += l0_idx * L1L2_PER_SELECT_UPPER;
            // need to know *exact* rank of position at start of current lower block
            // because our l1 indices store at max 2 ** 18, meaning one select upper
            // block can contain multiple regular upper level blocks and the l1 value
            // is not necessarily a true cumulative count of the rank within sel upper
            // the +1 is to ensure rank is run on the starting bit of the current block
            // (so the count will exclude its value)
            uint64_t full_rank = this->l0[l1l2_idx / LOWER_PER_UPPER] + (this->l1l2[l1l2_idx] >> EF_TOTAL_COUNT);
                //this->rank1(bv, (l1l2_idx * LOWER_BLOCK_COUNT) /*+ 1*/);
            uint64_t full_next_rank = full_rank;
            // limit (end of select upper block) is either: size of all l1l2 indices
            // or: last L2 index in current select upper block, whichever is lower
            uint64_t last_in_upper = (L1L2_PER_SELECT_UPPER * l0_idx) + L1L2_PER_SELECT_UPPER;
            uint64_t limit = std::min<uint64_t>(L1L2_INDEX_COUNT, last_in_upper);
            while ((l1l2_idx + 1) < limit) {
                full_next_rank = this->l0[(l1l2_idx + 1) / LOWER_PER_UPPER]
                        + (this->l1l2[l1l2_idx + 1] >> EF_TOTAL_COUNT);
                if (full_next_rank >= i) {
                    break;
                }
                full_rank = full_next_rank;
                ++l1l2_idx;
            }
            full_next_rank = full_rank;
            rank -= (full_rank - l0);
            // elias-fano scan and decode L2s
            full_rank = 0;
            uint64_t idx = 0;
            __uint128_t l1l2_entry = this->l1l2[l1l2_idx];
            __uint128_t l1l2_lower = l1l2_entry >> EF_UPPER_BV_COUNT;
            for (; idx < N_L2; ++idx) {
                uint64_t ef_upper_bits =
                    ((_tzcnt_u64(_pdep_u64(1ul << idx, l1l2_entry)) /*+ 1*/) - idx)
                    << EF_LOWER_ELE_COUNT;
                uint64_t ef_lower_bits = EF_LOWER_MASK & (l1l2_lower >> (idx * EF_LOWER_ELE_COUNT));
                uint64_t l2 = ef_lower_bits | ef_upper_bits;
                uint64_t full_next_rank_2 = full_next_rank + l2;
                if (full_next_rank_2 >= i) {
                    break;
                }
                full_rank = l2;
            }
            rank -= full_rank;
            // select within basic block
            // start_position is of first word in bb
            uint64_t start_position = (l1l2_idx * LOWER_BLOCK_WORDS) + (idx * BASIC_BLOCK_WORDS);
            uint64_t popc = 0;
            while ((popc = std::popcount<uint64_t>(bv[start_position])) < rank) {
                ++start_position;
                rank -= popc;
            }
            uint64_t in_word_result = _tzcnt_u64(_pdep_u64(1ul << (rank - 1), bv[start_position]));
            uint64_t final_result = (start_position * 64) + in_word_result;
            return final_result;
        }
        
        void print(size_t max_l0 = ULONG_MAX, size_t max_l1l2 = ULONG_MAX) {
            size_t l0_count = this->bv_count / UPPER_BLOCK_COUNT;
            size_t l0_len = l0_count + 1;
            max_l0 = std::min(l0_len, max_l0);
            size_t num_lower_blocks = this->bv_count / LOWER_BLOCK_COUNT;
            if (this->bv_count % LOWER_BLOCK_COUNT) num_lower_blocks++;
            max_l1l2 = std::min(num_lower_blocks, max_l1l2);
            cout << "=== l0 index === " << endl;
            for (size_t idx = 0; idx < max_l0; idx++) {
                auto x = this->l0[idx];
                cout << x << ", ";
                print_bits<uint64_t>(x);
                cout << endl;
            }
            for (
                size_t idx = 0;
                idx < max_l1l2;
                idx++
            ) {
                __uint128_t l1l2 = this->l1l2[idx];
                uint64_t l1 = (uint64_t) (l1l2 >> EF_TOTAL_COUNT);
                cout << "*** l1 index " << idx << " ***" << endl;
                cout << l1 << ", ";
                print_bits<uint64_t>(l1);
                cout << endl;
                cout << "--- l2 indices ---" << endl;
                cout << "$$$ ef upper $$$" << endl;
                uint32_t upper = (uint32_t) (l1l2 & EF_UPPER_BV_MASK);
                print_bits<uint32_t>(upper, EF_UPPER_BV_COUNT);
                cout << endl;
                cout << "... ef lower ..." << endl;
                for (size_t j = 0; j < 10; j++) {
                    uint16_t lower = (uint16_t) (l1l2 >> (EF_UPPER_BV_COUNT + (j * EF_LOWER_ELE_COUNT)));
                    print_bits<uint16_t>(lower, 9);
                    cout << endl;
                }
            }

        }

};

#endif /* ORZO_H */
