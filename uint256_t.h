/*
uint256_t.h
An unsigned 256 bit integer library for C++
By Jason Lee @ calccrypto@yahoo.com
with much help from Auston Sterling

From http://calccrypto.wikidot.com/programming:uint256-t
Licenced http://creativecommons.org/licenses/by-sa/3.0/
*/

#pragma once

#include <cstdlib>
#include <iostream>
#include <stdint.h>

#include "uint128_t.h"

class uint256_t{
	private:
		uint128_t UPPER, LOWER;

	public:
		// Constructors
		uint256_t(){
			UPPER = 0;
			LOWER = 0;
		}

		template <typename T>
		uint256_t(T rhs){
			UPPER = 0;
			LOWER = (uint128_t) rhs;
		}

		template <typename S, typename T>
		uint256_t(const S upper_rhs, const T lower_rhs){
			UPPER = (uint128_t) upper_rhs;
			LOWER = (uint128_t) lower_rhs;
		}

		uint256_t(const uint256_t & rhs){
			UPPER = rhs.UPPER;
			LOWER = rhs.LOWER;
		}

		//  RHS input args only

		// Assignment Operator
		template <typename T> uint256_t operator=(T rhs){
			UPPER = 0;
			LOWER = (uint128_t) rhs;
			return *this;
		}

		uint256_t operator=(uint256_t rhs){
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
			return (uint64_t) LOWER;
		}

		operator uint128_t(){
			return LOWER;
		}

		// Bitwise Operators
		template <typename T> uint256_t operator&(T rhs){
			return uint256_t(0, LOWER & (uint128_t) rhs);
		}

		uint256_t operator&(uint256_t rhs){
			return uint256_t(UPPER & rhs.UPPER, LOWER & rhs.LOWER);
		}

		template <typename T> uint256_t operator|(T rhs){
			return uint256_t(UPPER, LOWER | uint128_t(rhs));
		}

		uint256_t operator|(uint256_t rhs){
			return uint256_t(UPPER | rhs.UPPER, LOWER | rhs.LOWER);
		}

		template <typename T> uint256_t operator^(T rhs){
			return uint256_t(UPPER, LOWER ^ (uint128_t) rhs);
		}

		uint256_t operator^(uint256_t rhs){
			return uint256_t(UPPER ^ rhs.UPPER, LOWER ^ rhs.LOWER);
		}

		template <typename T> uint256_t operator&=(T rhs){
			UPPER = 0;
			LOWER &= rhs;
			return *this;
		}

		uint256_t operator&=(uint256_t rhs){
			UPPER &= rhs.UPPER;
			LOWER &= rhs.LOWER;
			return *this;
		}

		template <typename T> uint256_t operator|=(T rhs){
			LOWER |= (uint128_t) rhs;
			return *this;
		}

		uint256_t operator|=(uint256_t rhs){
			UPPER |= rhs.UPPER;
			LOWER |= rhs.LOWER;
			return *this;
		}

		template <typename T> uint256_t operator^=(T rhs){
			LOWER ^= (uint128_t) rhs;
			return *this;
		}

		uint256_t operator^=(const uint256_t rhs){
			UPPER ^= rhs.UPPER;
			LOWER ^= rhs.LOWER;
			return *this;
		}

		uint256_t operator~(){
			return uint256_t(~UPPER, ~LOWER);
		}

		// Bit Shift Operators
		uint256_t operator<<(int shift){
			if (shift >= 256)
				return uint256_t(0, 0);
			else if (shift == 128)
				return uint256_t(LOWER, 0);
			else if (shift == 0)
				return *this;
			else if (shift < 128)
				return uint256_t((UPPER << shift) + (LOWER >> (128 - shift)), LOWER << shift);
			else if ((256 > shift) && (shift > 128))
				return uint256_t(LOWER << (shift - 128), 0);
			else
				return uint256_t(0);
		}

		template <typename T>
		uint256_t operator>>(const T shift){
			if (shift >= 256)
				return uint256_t(0, 0);
			else if (shift == 128)
				return uint256_t(0, UPPER);
			else if (shift == 0)
				return *this;
			else if (shift < 128)
				return uint256_t(UPPER >> shift, (UPPER << (128 - shift)) + (LOWER >> shift));
			else if ((256 > shift) && (shift > 128))
				return uint256_t(0, (UPPER >> (shift - 128)));
			else
				return uint256_t(0);
		}

		uint256_t operator<<=(const int shift){
			*this = *this << shift;
			return *this;
		}

		uint256_t operator>>=(int shift){
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

		template <typename T> bool operator&&(uint256_t rhs){
			return (bool) *this && (bool) rhs;
		}

		template <typename T> bool operator||(T rhs){
			return ((bool) *this) || rhs;

		}

		template <typename T> bool operator||(uint256_t rhs){
			return ((bool) *this) || (bool) rhs;
		}

		// Comparison Operators
		template <typename T> bool operator==(T rhs){
			return (!UPPER && (LOWER == uint128_t(rhs)));
		}

		bool operator==(uint256_t rhs){
			return ((UPPER == rhs.UPPER) && (LOWER == rhs.LOWER));
		}

		template <typename T> bool operator!=(T rhs){
			return (UPPER | (LOWER != (uint128_t) rhs));
		}

		bool operator==(uint128_t rhs){
			return (!UPPER && (LOWER == rhs));
		}

		bool operator!=(uint256_t rhs){
			return ((UPPER != rhs.UPPER) | (LOWER != rhs.LOWER));
		}

		template <typename T> bool operator>(T rhs){
			if (UPPER)
				return true;
			return (LOWER > (uint128_t) rhs);
		}

		bool operator>(uint256_t rhs){
			if (UPPER == rhs.UPPER)
				return (LOWER > rhs.LOWER);
			if (UPPER > rhs.UPPER)
				return true;
			return false;
		}

		template <typename T> bool operator<(T rhs){
			if  (!UPPER)
				return (LOWER < (uint128_t) rhs);
			return false;
		}

		bool operator<(uint256_t rhs){
			if (UPPER == rhs.UPPER)
				return (LOWER < rhs.LOWER);
			if (UPPER < rhs.UPPER)
				return true;
			return false;
		}

		template <typename T> bool operator>=(T rhs){
			return ((*this > rhs) | (*this == rhs));
		}

		bool operator>=(uint256_t rhs){
			return ((*this > rhs) | (*this == rhs));
		}

		template <typename T> bool operator<=(T rhs){
			return ((*this < rhs) | (*this == rhs));
		}

		bool operator<=(uint256_t rhs){
			return ((*this < rhs) | (*this == rhs));
		}

		// Arithmetic Operators
		template <typename T> uint256_t operator+(T rhs){
			return uint256_t(UPPER + ((LOWER + (uint128_t) rhs) < LOWER), LOWER + (uint128_t) rhs);
		}

		uint256_t operator+(uint256_t rhs){
			return uint256_t(rhs.UPPER + UPPER + ((LOWER + rhs.LOWER) < LOWER), LOWER + rhs.LOWER);
		}

		template <typename T> uint256_t operator+=(T rhs){
			UPPER = UPPER + ((LOWER + rhs) < LOWER);
			LOWER = LOWER + rhs;
			return *this;
		}

		uint256_t operator+=(uint256_t rhs){
			UPPER = rhs.UPPER + UPPER + ((LOWER + rhs.LOWER) < LOWER);
			LOWER = LOWER + rhs.LOWER;
			return *this;
		}

		template <typename T> uint256_t  operator-(T rhs){
			return uint256_t(UPPER - ((LOWER - rhs) > LOWER), LOWER - rhs);
		}

		uint256_t  operator-(uint256_t rhs){
			return uint256_t(UPPER - rhs.UPPER - ((LOWER - rhs.LOWER) > LOWER), LOWER - rhs.LOWER);;
		}

		template <typename T> uint256_t operator-=(T rhs){
			*this = *this - rhs;
			return *this;
		}

		uint256_t operator-=(uint256_t rhs){
			*this = *this - rhs;
			return *this;
		}

		template <typename T> uint256_t operator*(T rhs){
			return *this * uint256_t(rhs);
		}

		uint256_t operator*(uint256_t rhs){
			// split values into 4 64-bit parts
			uint128_t top[4] = {UPPER >> 64, UPPER % uint128_t(1, 0), LOWER >> 64, LOWER % uint128_t(1, 0)};
			uint128_t bottom[4] = {rhs.upper() >> 64, rhs.upper() % uint128_t(1, 0), rhs.lower() >> 64, rhs.lower() % uint128_t(1, 0)};
			uint128_t products[4][4];

			for(int y = 3; y > -1; y--)
				for(int x = 3; x > -1; x--){
					products[3 - x][y] = top[x] * bottom[y];
			}

			// initial row
			uint128_t fourth64 = products[0][3] % uint128_t(1, 0);
			uint128_t third64 = products[0][2] % uint128_t(1, 0) + (products[0][3] >> 64);
			uint128_t second64 = products[0][1] % uint128_t(1, 0) + (products[0][2] >> 64);
			uint128_t first64 = products[0][0] % uint128_t(1, 0) + (products[0][1] >> 64);

			// second row
			third64 += products[1][3] % uint128_t(1, 0);
			second64 += (products[1][2] % uint128_t(1, 0)) + (products[1][3] >> 64);
			first64 += (products[1][1] % uint128_t(1, 0)) + (products[1][2] >> 64);

			// third row
			second64 += products[2][3] % uint128_t(1, 0);
			first64 += (products[2][2] % uint128_t(1, 0)) + (products[2][3] >> 64);

			// fourth row
			first64 += products[3][3] % uint128_t(1, 0);

			// combines the values, taking care of carry over
			return uint256_t(first64 << 64, 0) + uint256_t(third64 >> 64, third64 << 64) + uint256_t(second64, 0) + uint256_t(fourth64);
		}

		template <typename T> uint256_t operator*=(T rhs){
			*this = *this * uint256_t(rhs);
			return *this;
		}

		uint256_t operator*=(uint256_t rhs){
			*this = *this * rhs;
			return *this;
		}

		template <typename T> uint256_t operator/(T rhs){
			return *this / uint256_t(rhs);
		}

		uint256_t operator/(uint256_t rhs){
			// Save some calculations //////////////////////
			if (rhs == 0){
				std::cout << "Error: division or modulus by zero" << std::endl;
				exit(1);
			}
			if (rhs == 1)
				return *this;
			if (*this == rhs)
				return uint256_t(1);
			if ((*this == 0) | (*this < rhs))
				return uint256_t(0);
			// Checks for divisors that are powers of two
			uint16_t s = 0;
			uint256_t copyd(rhs);
			while ((copyd.LOWER & 1) == 0){
				copyd >>= 1;
				s++;
			}
			if (copyd == 1)
				return *this >> s;
			////////////////////////////////////////////////
			uint256_t copyn(*this), quotient = 0;
			while (copyn >= rhs){
				uint256_t copyd(rhs), temp(1);
				// shift the divosr to match the highest bit
				while ((copyn >> 1) > copyd){
					copyd <<= 1;
					temp <<= 1;
				}
				copyn -= copyd;
				quotient += temp;
			}
			return quotient;
		}

		template <typename T> uint256_t operator/=(T rhs){
			*this = *this / uint256_t(rhs);
			return *this;
		}

		uint256_t operator/=(uint256_t rhs){
			*this = *this / rhs;
			return *this;
		}

		template <typename T> uint256_t operator%(T rhs){
			return *this % uint256_t(rhs);
		}

		uint256_t operator%(uint256_t rhs){
			return *this - (rhs * (*this / rhs));
		}

		template <typename T> uint256_t operator%=(T rhs){
			*this = *this % uint256_t(rhs);
			return *this;
		}

		uint256_t operator%=(uint256_t rhs){
			*this = *this % rhs;
			return *this;
		}

		// Increment Operators
		uint256_t operator++(){
			*this += 1;
			return *this;
		}

		uint256_t operator++(int){
			uint256_t temp(*this);
			++*this;
			return temp;
		}

		// Decrement Operators
		uint256_t operator--(){
			*this -= 1;
			return *this;
		}

		uint256_t operator--(int){
			uint256_t temp(*this);
			--*this;
			return temp;
		}

		// get private values
		uint128_t upper(){
			return UPPER;
		}

		uint128_t lower(){
			return LOWER;
		}
};
// lhs type T as first arguemnt

// Bitwise Operators
template <typename T> T operator&(T lhs, uint256_t rhs){
	T out = lhs & (T) rhs.lower();
	return out;
}

template <typename T> T operator|(T lhs, uint256_t rhs){
	T out = lhs | (T) rhs.lower();
	return out;
}

template <typename T> T operator^(T lhs, uint256_t rhs){
	T out = lhs ^ (T) rhs.lower();
	return out;
}

template <typename T> T operator&=(T & lhs, uint256_t rhs){
	lhs &= (T) rhs.lower();
	return lhs;
}

template <typename T> T operator|=(T & lhs, uint256_t rhs){
	lhs |= (T) rhs.lower();
	return lhs;
}

template <typename T> T operator^=(T & lhs, uint256_t rhs){
	lhs ^= (T) rhs.lower();
	return lhs;
}

// Comparison Operators
template <typename T> bool operator==(T lhs, uint256_t rhs){
	return (!rhs.upper() && (uint128_t) lhs == rhs.lower());
}

template <typename T> bool operator!=(T lhs, uint256_t rhs){
	return (rhs.upper() | ((uint128_t) lhs != rhs.lower()));
}

template <typename T> bool operator>(T lhs, uint256_t rhs){
	if (rhs.upper())
		return false;
	return ((uint128_t) lhs > rhs.lower());
}

template <typename T> bool operator<(T lhs, uint256_t rhs){
	if (rhs.upper())
		return true;
	return ((uint128_t) lhs < rhs.lower());
}

template <typename T> bool operator>=(T lhs, uint256_t rhs){
	if (rhs.upper())
		return false;
	return ((uint128_t) lhs >= rhs.lower());
}

template <typename T> bool operator<=(T lhs, uint256_t rhs){
	if (rhs.upper())
		return true;
	return ((uint128_t) lhs <= rhs.lower());
}

// Arithmetic Operators
template <typename T> T operator+(T lhs, uint256_t rhs){
	return (T) (rhs + lhs);
}

template <typename T> T & operator+=(T & lhs, uint256_t rhs){
	lhs = (T) (rhs + lhs);
	return lhs;
}

template <typename T> T operator-(T lhs, uint256_t rhs){
	return (T) (rhs - lhs);
}

template <typename T> T & operator-=(T & lhs, uint256_t rhs){
	lhs = (T) (rhs - lhs);
	return lhs;
}

template <typename T> T operator*(T lhs, uint256_t rhs){
	return lhs * rhs.lower();
}

template <typename T> T & operator*=(T & lhs, uint256_t rhs){
	lhs = (T) (rhs.lower() * lhs);
	return lhs;
}

template <typename T> T operator/(T lhs, uint256_t rhs){
	return (T) (uint256_t(lhs) / rhs);
}

template <typename T> T & operator/=(T & lhs, uint256_t rhs){
	lhs = (T) (uint256_t(lhs) / rhs);
	return lhs;
}

template <typename T> T operator%(T lhs, uint256_t rhs){
	return (T) (uint256_t(lhs) % rhs);
}

template <typename T> T & operator%=(T & lhs, uint256_t rhs){
	lhs = (T) (uint256_t(lhs) % rhs);
	return lhs;
}

// IO Operator
inline std::ostream & operator<<(std::ostream & stream, uint256_t rhs){
	std::string out = "";
	if (rhs == 0)
		out = "0";
	else {
		int div;
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
