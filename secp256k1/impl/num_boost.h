// Copyright (c) 2014 Tim Hughes
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef _SECP256K1_NUM_REPR_IMPL_H_
#define _SECP256K1_NUM_REPR_IMPL_H_
#include <assert.h>
#include <boost/math/common_factor.hpp>

void static secp256k1_num_init(secp256k1_num_t *r)
{
	*r = 0;
}

void static secp256k1_num_free(secp256k1_num_t*)
{
}

void static secp256k1_num_copy(secp256k1_num_t *r, const secp256k1_num_t *a)
{
	*r = *a;
}

int static secp256k1_num_bits(const secp256k1_num_t *a)
{
	int numLimbs = a->backend().size();
    int ret = (numLimbs - 1) * a->backend().limb_bits;
    for (auto x = a->backend().limbs()[numLimbs - 1]; x; x >>= 1, ++ret);
    return ret;
}

void static secp256k1_num_get_bin(unsigned char *r, unsigned int rlen, const secp256k1_num_t *a)
{
	for (auto n = abs(*a); n; n >>= 8)
	{
		assert(rlen > 0); // out of space?
		r[--rlen] = n.convert_to<unsigned char>();
	}
	memset(r, 0, rlen);
}

void static secp256k1_num_set_bin(secp256k1_num_t *r, const unsigned char *a, unsigned int alen)
{
	*r = 0;
	for (unsigned int i = 0; i != alen; ++i)
	{
		*r <<= 8;
		*r |= a[i];
	}
}

void static secp256k1_num_set_int(secp256k1_num_t *r, int a)
{
    *r = a;
}

void static secp256k1_num_mod(secp256k1_num_t *r, const secp256k1_num_t *m)
{
	*r %= *m;
}

void static secp256k1_num_mod_inverse(secp256k1_num_t *r, const secp256k1_num_t *n, const secp256k1_num_t *m)
{
	// http://rosettacode.org/wiki/Modular_inverse
	secp256k1_num_t a = *n;
	secp256k1_num_t b = *m;
	secp256k1_num_t x0 = 0;
	secp256k1_num_t x1 = 1;
	assert(*n > 0);
	assert(*m > 0);
	if (b != 1)
	{
		secp256k1_num_t q, t;
		while (a > 1)
		{
			boost::multiprecision::divide_qr(a, b, q, t);
			a = b; b = t;

			t = x1 - q * x0;
			x1 = x0; x0 = t;
		}
		if (x1 < 0)
		{
			x1 += *m;
		}
	}
	*r = x1;

	// check result
	#ifdef _DEBUG
	{
		typedef boost::multiprecision::number<boost::multiprecision::cpp_int_backend<512, 512, boost::multiprecision::signed_magnitude, boost::multiprecision::unchecked, void>> bignum;
		bignum br = *r, bn = *n, bm = *m;
		assert((((bn) * (br)) % bm) == 1);
	}
	#endif
}

int static secp256k1_num_is_zero(const secp256k1_num_t *a)
{
    return a->is_zero();
}

int static secp256k1_num_is_odd(const secp256k1_num_t *a)
{
    return boost::multiprecision::bit_test(*a, 0);
}

int static secp256k1_num_is_neg(const secp256k1_num_t *a)
{
	return a->backend().isneg();
}

int static secp256k1_num_cmp(const secp256k1_num_t *a, const secp256k1_num_t *b)
{
	return a->backend().compare_unsigned(b->backend());
}

void static secp256k1_num_add(secp256k1_num_t *r, const secp256k1_num_t *a, const secp256k1_num_t *b)
{
	*r = (*a) + (*b);
}

void static secp256k1_num_sub(secp256k1_num_t *r, const secp256k1_num_t *a, const secp256k1_num_t *b)
{
	*r = (*a) - (*b);
}

void static secp256k1_num_mul(secp256k1_num_t *r, const secp256k1_num_t *a, const secp256k1_num_t *b)
{
	*r = (*a) * (*b);
}

void static secp256k1_num_div(secp256k1_num_t *r, const secp256k1_num_t *a, const secp256k1_num_t *b)
{
	*r = (*a) / (*b);
}

void static secp256k1_num_mod_mul(secp256k1_num_t *r, const secp256k1_num_t *a, const secp256k1_num_t *b, const secp256k1_num_t *m)
{
    secp256k1_num_mul(r, a, b);
    secp256k1_num_mod(r, m);
}

int static secp256k1_num_shift(secp256k1_num_t *r, int bits)
{
	unsigned ret = r->convert_to<unsigned>() & ((1 << bits) - 1);
	*r >>= bits;
	return ret;
}

int static secp256k1_num_get_bit(const secp256k1_num_t *a, int pos)
{
	return boost::multiprecision::bit_test(*a, pos);
}

void static secp256k1_num_inc(secp256k1_num_t *r)
{
	++*r;
}

void static secp256k1_num_set_hex(secp256k1_num_t *r, const char *a, int alen)
{
    static const unsigned char cvt[256] = {
        0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
        0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
        0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
        0, 1, 2, 3, 4, 5, 6,7,8,9,0,0,0,0,0,0,
        0,10,11,12,13,14,15,0,0,0,0,0,0,0,0,0,
        0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
        0,10,11,12,13,14,15,0,0,0,0,0,0,0,0,0,
        0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
        0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
        0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
        0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
        0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
        0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
        0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
        0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
        0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0
    };
	*r = 0;
	for (int i = 0; i != alen; ++i)
	{
		*r <<= 4;
		*r |= cvt[a[i]];
	}
}

void static secp256k1_num_get_hex(char *r, int rlen, const secp256k1_num_t *a)
{
    static const unsigned char cvt[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
	for (auto n = *a; n; n >>= 4)
	{
		assert(rlen > 0); // out of space?
		r[--rlen] = cvt[n.convert_to<unsigned char>() & 15];
	}
	memset(r, '0', rlen);
}

void static secp256k1_num_split(secp256k1_num_t *rl, secp256k1_num_t *rh, const secp256k1_num_t *a, int bits)
{
	*rl = *a & ((secp256k1_num_t(1) << bits) - 1);
	*rh = *a >> bits;
}

void static secp256k1_num_negate(secp256k1_num_t *r)
{
	r->backend().negate();
}

#endif
