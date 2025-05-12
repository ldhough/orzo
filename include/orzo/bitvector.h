#ifndef BITVECTOR_H
#define BITVECTOR_H

#include <cassert>
#include <iostream>
#include <cstring>

using std::cout, std::endl;

class OrzoBitvector {

    private:

        uint64_t *bv;

    public:

        OrzoBitvector(
            uint64_t n,
            uint64_t multiple_of, // in bits
            uint64_t alignment = 64
        ) {
            uint64_t *bv;
            uint64_t num_words = n / 64;
            uint64_t mow = multiple_of / 64; // multiple of in words
            num_words += (num_words % mow);
            bv = (uint64_t*) aligned_alloc(alignment, num_words * sizeof(*bv));
            memset(bv, 0, num_words * sizeof(*bv));
#ifdef DEBUG
            bool divisible = (((uint64_t) bv) % 64) == 0;
            cout << "bv"
                << ((divisible) ? " is aligned" : " is not aligned")
                << endl;
#endif
            this->bv = bv;
        }

        ~OrzoBitvector() {
            free(this->bv);
        }

        uint64_t *data() {
            return this->bv;
        }

        void set_bit(uint64_t i) {
            uint64_t word_idx = i / 64;
            this->bv[word_idx] |= (1ul << (i % 64));
        }

        bool get_bit(uint64_t i) {
            uint64_t word_idx = i / 64;
            return (bool) (this->bv[word_idx] & (1 << (i % 64)));
        }

};

#endif /* BITVECTOR_H */
