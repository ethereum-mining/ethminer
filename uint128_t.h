/*
uint128_t.h
An unsigned 128 bit integer library for C++
By Jason Lee @ calccrypto@yahoo.com
with much help from Auston Sterling

And thanks to Stefan Deigmüller for finding
a bug in operator*.

From http://calccrypto.wikidot.com/programming:uint128-t
Licenced http://creativecommons.org/licenses/by-sa/3.0/
*/

#pragma once

#include <cstdlib>
#include <iostream>
#include <stdint.h>

class uint128_t{
	private:
		uint64_t UPPER, LOWER;

	public:
		// Constructors
		uint128_t(){
			UPPER = 0;
			LOWER = 0;
		}

		template <typename T>
		uint128_t(T rhs){
			UPPER = 0;
			LOWER = (uint64_t) rhs;
		}

		template <typename S, typename T>
		uint128_t(const S upper_rhs, const T lower_rhs){
			UPPER = (uint64_t) upper_rhs;
			LOWER = (uint64_t) lower_rhs;
		}

		uint128_t(const uint128_t & rhs){
			UPPER = rhs.UPPER;
			LOWER = rhs.LOWER;
		}

		//  RHS input args only

		// Assignment Operator
		template <typename T> uint128_t operator=(T rhs){
			UPPER = 0;
			LOWER = (uint64_t) rhs;
			return *this;
		}

		uint128_t operator=(uint128_t rhs){
			UPPER = rhs.UPPER;
			LOWER = rhs.LOWER;
			return *this;
		}

		// Typecast Operators
		operator bool(){
			return (bool) (UPPER | LOWER);
		}

		operator char(){
			return (char) LOWER;
		}

		operator int(){
			return (int) LOWER;
		}

		operator uint8_t(){
			return (uint8_t) LOWER;
		}

		operator uint16_t(){
			return (uint16_t) LOWER;
		}

		operator uint32_t(){
			return (uint32_t) LOWER;
		}

		operator uint64_t(){
			return LOWER;
		}

		// Bitwise Operators
		template <typename T> uint128_t operator&(T rhs){
			return uint128_t(0, LOWER & (uint64_t) rhs);
		}

		uint128_t operator&(uint128_t rhs){
			return uint128_t(UPPER & rhs.UPPER, LOWER & rhs.LOWER);
		}

		template <typename T> uint128_t operator|(T rhs){
			return uint128_t(UPPER, LOWER | (uint64_t) rhs);
		}

		uint128_t operator|(uint128_t rhs){
			return uint128_t(UPPER | rhs.UPPER, LOWER | rhs.LOWER);
		}

		template <typename T> uint128_t operator^(T rhs){
			return uint128_t(UPPER, LOWER ^ (uint64_t) rhs);
		}

		uint128_t operator^(uint128_t rhs){
			return uint128_t(UPPER ^ rhs.UPPER, LOWER ^ rhs.LOWER);
		}

		template <typename T> uint128_t operator&=(T rhs){
			UPPER = 0;
			LOWER &= rhs;
			return *this;
		}

		uint128_t operator&=(uint128_t rhs){
			UPPER &= rhs.UPPER;
			LOWER &= rhs.LOWER;
			return *this;
		}

		template <typename T> uint128_t operator|=(T rhs){
			LOWER |= (uint64_t) rhs;
			return *this;
		}

		uint128_t operator|=(uint128_t rhs){
			UPPER |= rhs.UPPER;
			LOWER |= rhs.LOWER;
			return *this;
		}

		template <typename T> uint128_t operator^=(T rhs){
			LOWER ^= (uint64_t) rhs;
			return *this;
		}

		uint128_t operator^=(const uint128_t rhs){
			UPPER ^= rhs.UPPER;
			LOWER ^= rhs.LOWER;
			return *this;
		}

		uint128_t operator~(){
			return uint128_t(~UPPER, ~LOWER);
		}

		// Bit Shift Operators
		template <typename T>
		uint128_t operator<<(const T shift){
			if (shift >= 128)
				return uint128_t(0, 0);
			else if (shift == 64)
				return uint128_t(LOWER, 0);
			else if (shift == 0)
				return *this;
			else if (shift < 64)
				return uint128_t((UPPER << shift) + (LOWER >> (64 - shift)), LOWER << shift);
			else if ((128 > shift) && (shift > 64))
				return uint128_t(LOWER << (shift - 64), 0);
			else
				return uint128_t(0);
		}

		template <typename T>
		uint128_t operator>>(const T shift){
			if (shift >= 128)
				return uint128_t(0, 0);
			else if (shift == 64)
				return uint128_t(0, UPPER);
			else if (shift == 0)
				return *this;
			else if (shift < 64)
				return uint128_t(UPPER >> shift, (UPPER << (64 - shift)) + (LOWER >> shift));
			else if ((128 > shift) && (shift > 64))
				return uint128_t(0, (UPPER >> (shift - 64)));
			else
				return uint128_t(0);
		}

		uint128_t operator<<=(int shift){
			*this = *this << shift;
			return *this;
		}

		uint128_t operator>>=(int shift){
			*this = *this >> shift;
			return *this;
		}

		// Logical Operators
		bool operator!(){
			return !(bool) (UPPER | LOWER);
		}

		template <typename T> bool operator&&(T rhs){
			return (bool) *this && rhs;
		}

		template <typename T> bool operator&&(uint128_t rhs){
			return (bool) *this && (bool) rhs;
		}

		template <typename T> bool operator||(T rhs){
			return ((bool) *this) || rhs;

		}

		template <typename T> bool operator||(uint128_t rhs){
			return ((bool) *this) || (bool) rhs;
		}

		// Comparison Operators
		template <typename T> bool operator==(T rhs){
			return (!UPPER && (LOWER == (uint64_t) rhs));
		}

		bool operator==(uint128_t rhs){
			return ((UPPER == rhs.UPPER) && (LOWER == rhs.LOWER));
		}

		template <typename T> bool operator!=(T rhs){
			return (UPPER | (LOWER != (uint64_t) rhs));
		}

		bool operator!=(uint128_t rhs){
			return ((UPPER != rhs.UPPER) | (LOWER != rhs.LOWER));
		}

		template <typename T> bool operator>(T rhs){
			if (UPPER)
				return true;
			return (LOWER > (uint64_t) rhs);
		}

		bool operator>(uint128_t rhs){
			if (UPPER == rhs.UPPER)
				return (LOWER > rhs.LOWER);
			if (UPPER > rhs.UPPER)
				return true;
			return false;
		}

		template <typename T> bool operator<(T rhs){
			if  (!UPPER)
				return (LOWER < (uint64_t) rhs);
			return false;
		}

		bool operator<(uint128_t rhs){
			if (UPPER == rhs.UPPER)
				return (LOWER < rhs.LOWER);
			if (UPPER < rhs.UPPER)
				return true;
			return false;
		}

		template <typename T> bool operator>=(T rhs){
			return ((*this > rhs) | (*this == rhs));
		}

		bool operator>=(uint128_t rhs){
			return ((*this > rhs) | (*this == rhs));
		}

		template <typename T> bool operator<=(T rhs){
			return ((*this < rhs) | (*this == rhs));
		}

		bool operator<=(uint128_t rhs){
			return ((*this < rhs) | (*this == rhs));
		}

		// Arithmetic Operators
		template <typename T> uint128_t operator+(T rhs){
			return uint128_t(UPPER + ((LOWER + (uint64_t) rhs) < LOWER), LOWER + (uint64_t) rhs);
		}

		uint128_t operator+(uint128_t rhs){
			return uint128_t(rhs.UPPER + UPPER + ((LOWER + rhs.LOWER) < LOWER), LOWER + rhs.LOWER);
		}

		template <typename T> uint128_t operator+=(T rhs){
			UPPER = UPPER + ((LOWER + rhs) < LOWER);
			LOWER = LOWER + rhs;
			return *this;
		}

		uint128_t operator+=(uint128_t rhs){
			UPPER = rhs.UPPER + UPPER + ((LOWER + rhs.LOWER) < LOWER);
			LOWER = LOWER + rhs.LOWER;
			return *this;
		}

		template <typename T> uint128_t  operator-(T rhs){
			return uint128_t((uint64_t) (UPPER - ((LOWER - rhs) > LOWER)), (uint64_t) (LOWER - rhs));
		}

		uint128_t  operator-(uint128_t rhs){
			return uint128_t(UPPER - rhs.UPPER - ((LOWER - rhs.LOWER) > LOWER), LOWER - rhs.LOWER);
		}

		template <typename T> uint128_t operator-=(T rhs){
			*this = *this - rhs;
			return *this;
		}

		uint128_t operator-=(uint128_t rhs){
			*this = *this - rhs;
			return *this;
		}

		template <typename T> uint128_t operator*(T rhs){
			return *this * uint128_t(rhs);
		}

		uint128_t operator*(uint128_t rhs){
			// split values into 4 32-bit parts
			uint64_t top[4] = {UPPER >> 32, UPPER % 0x100000000ULL, LOWER >> 32, LOWER % 0x100000000ULL};
			uint64_t bottom[4] = {rhs.upper() >> 32, rhs.upper() % 0x100000000ULL, rhs.lower() >> 32, rhs.lower() % 0x100000000ULL};
			uint64_t products[4][4];

			for(int y = 3; y > -1; y--)
				for(int x = 3; x > -1; x--){
					products[3 - x][y] = top[x] * bottom[y];
			}

			// initial row
			uint64_t fourth32 = products[0][3] % 0x100000000ULL;
			uint64_t third32 = products[0][2] % 0x100000000ULL + (products[0][3] >> 32);
			uint64_t second32 = products[0][1] % 0x100000000ULL + (products[0][2] >> 32);
			uint64_t first32 = products[0][0] % 0x100000000ULL + (products[0][1] >> 32);

			// second row
			third32 += products[1][3] % 0x100000000ULL;
			second32 += (products[1][2] % 0x100000000ULL) + (products[1][3] >> 32);
			first32 += (products[1][1] % 0x100000000ULL) + (products[1][2] >> 32);

			// third row
			second32 += products[2][3] % 0x100000000ULL;
			first32 += (products[2][2] % 0x100000000ULL) + (products[2][3] >> 32);

			// fourth row
			first32 += products[3][3] % 0x100000000ULL;

			// combines the values, taking care of carry over
			return uint128_t(first32 << 32, 0) + uint128_t(third32 >> 32, third32 << 32) + uint128_t(second32, 0) + uint128_t(fourth32);
		}

		template <typename T> uint128_t operator*=(T rhs){
			*this = *this * uint128_t(rhs);
			return *this;
		}

		uint128_t operator*=(uint128_t rhs){
			*this = *this * rhs;
			return *this;
		}

		template <typename T> uint128_t operator/(T rhs){
			return *this / uint128_t(rhs);
		}

		uint128_t operator/(uint128_t rhs){
			// Save some calculations /////////////////////
			if (rhs == 0){
				std::cout << "Error: division or modulus by zero" << std::endl;
				exit(1);
			}
			if (rhs == 1)
				return *this;
			if (*this == rhs)
				return uint128_t(1);
			if ((*this == 0) | (*this < rhs))
				return uint128_t(0);
			// Checks for divisors that are powers of two
			uint16_t s = 0;
			uint128_t copyd(rhs);
			while ((copyd.LOWER & 1) == 0){
				copyd >>= 1;
				s++;
			}
			if (copyd == 1)
				return *this >> s;
			////////////////////////////////////////////////
			uint128_t copyn(*this), quotient = 0;
			while (copyn >= rhs){
				uint128_t copyd(rhs), temp(1);
				// shift the divsor to match the highest bit
				while ((copyn >> 1) > copyd){
					copyd <<= 1;
					temp <<= 1;
				}
				copyn -= copyd;
				quotient += temp;
			}
			return quotient;
		}

		template <typename T> uint128_t operator/=(T rhs){
			*this = *this / uint128_t(rhs);
			return *this;
		}

		uint128_t operator/=(uint128_t rhs){
			*this = *this / rhs;
			return *this;
		}

		template <typename T> uint128_t operator%(T rhs){
			return *this - (rhs * (*this / rhs));
		}

		uint128_t operator%(uint128_t rhs){
			return *this - (rhs * (*this / rhs));
		}

		template <typename T> uint128_t operator%=(T rhs){
			*this = *this % uint128_t(rhs);
			return *this;
		}

		uint128_t operator%=(uint128_t rhs){
			*this = *this % rhs;
			return *this;
		}

		// Increment Operator
		uint128_t operator++(){
			*this += 1;
			return *this;
		}

		uint128_t operator++(int){
			uint128_t temp(*this);
			++*this;
			return temp;
		}

		// Decrement Operator
		uint128_t operator--(){
			*this -= 1;
			return *this;
		}

		uint128_t operator--(int){
			uint128_t temp(*this);
			--*this;
			return temp;
		}

		// get private values
		uint64_t upper(){
			return UPPER;
		}

		uint64_t lower(){
			return LOWER;
		}
};
// lhs type T as first arguemnt

// Bitwise Operators
template <typename T> T operator&(T lhs, uint128_t rhs){
	T out = lhs & (T) rhs.lower();
	return out;
}

template <typename T> T operator|(T lhs, uint128_t rhs){
	T out = lhs | (T) rhs.lower();
	return out;
}

template <typename T> T operator^(T lhs, uint128_t rhs){
	T out = lhs ^ (T) rhs.lower();
	return out;
}

template <typename T> T operator&=(T & lhs, uint128_t rhs){
	lhs &= (T) rhs.lower();
	return lhs;
}

template <typename T> T operator|=(T & lhs, uint128_t rhs){
	lhs |= (T) rhs.lower();
	return lhs;
}

template <typename T> T operator^=(T & lhs, uint128_t rhs){
	lhs ^= (T) rhs.lower();
	return lhs;
}

// Comparison Operators
template <typename T> bool operator==(T lhs, uint128_t rhs){
	return (!rhs.upper() && ((uint64_t) lhs == rhs.lower()));
}

template <typename T> bool operator!=(T lhs, uint128_t rhs){
	return (rhs.upper() | ((uint64_t) lhs != rhs.lower()));
}

template <typename T> bool operator>(T lhs, uint128_t rhs){
	if (rhs.upper())
		return false;
	return ((uint64_t) lhs > rhs.lower());
}

template <typename T> bool operator<(T lhs, uint128_t rhs){
	if (rhs.upper())
		return true;
	return ((uint64_t) lhs < rhs.lower());
}

template <typename T> bool operator>=(T lhs, uint128_t rhs){
	if (rhs.upper())
		return false;
	return ((uint64_t) lhs >= rhs.lower());
}

template <typename T> bool operator<=(T lhs, uint128_t rhs){
	if (rhs.upper())
		return true;
	return ((uint64_t) lhs <= rhs.lower());
}

// Arithmetic Operators
template <typename T> T operator+(T lhs, uint128_t rhs){
	return (T) (rhs + lhs);
}

template <typename T> T & operator+=(T & lhs, uint128_t rhs){
	lhs = (T) (rhs + lhs);
	return lhs;
}

template <typename T> T operator-(T lhs, uint128_t rhs){
	return (T) (rhs - lhs);
}

template <typename T> T & operator-=(T & lhs, uint128_t rhs){
	lhs = (T) (rhs - lhs);
	return lhs;
}

template <typename T> T operator*(T lhs, uint128_t rhs){
	return lhs * rhs.lower();
}

template <typename T> T & operator*=(T & lhs, uint128_t rhs){
	lhs = (T) (rhs.lower() * lhs);
	return lhs;
}

template <typename T> T operator/(T lhs, uint128_t rhs){
	return (T) (uint128_t(lhs) / rhs);
}

template <typename T> T & operator/=(T & lhs, uint128_t rhs){
	lhs = (T) (uint128_t(lhs) / rhs);
	return lhs;
}

template <typename T> T operator%(T lhs, uint128_t rhs){
	return (T) (uint128_t(lhs) % rhs);
}

template <typename T> T & operator%=(T & lhs, uint128_t rhs){
	lhs = (T) (uint128_t(lhs) % rhs);
	return lhs;
}

// IO Operator
inline std::ostream & operator<<(std::ostream & stream, uint128_t rhs){
	std::string out = "";
	if (rhs == 0)
		out = "0";
	else {
		int div = 10;
		if (stream.flags() & stream.oct)
			div = 8;
		if (stream.flags() & stream.dec)
			div = 10;
		if (stream.flags() & stream.hex)
			div = 16;
		while (rhs > 0){
			out = "0123456789abcdef"[size_t(rhs % div)] + out;
			rhs /= div;
		}
	}
	stream << out;
	return stream;
}
