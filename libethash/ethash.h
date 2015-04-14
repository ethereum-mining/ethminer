/*
  This file is part of ethash.

  ethash is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ethash is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ethash.  If not, see <http://www.gnu.org/licenses/>.
*/

/** @file ethash.h
* @date 2015
*/
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include "compiler.h"

#define ETHASH_REVISION 23
#define ETHASH_DATASET_BYTES_INIT 1073741824U // 2**30
#define ETHASH_DATASET_BYTES_GROWTH 8388608U  // 2**23
#define ETHASH_CACHE_BYTES_INIT 1073741824U // 2**24
#define ETHASH_CACHE_BYTES_GROWTH 131072U  // 2**17
#define ETHASH_EPOCH_LENGTH 30000U
#define ETHASH_MIX_BYTES 128
#define ETHASH_HASH_BYTES 64
#define ETHASH_DATASET_PARENTS 256
#define ETHASH_CACHE_ROUNDS 3
#define ETHASH_ACCESSES 64

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ethash_params {
	uint64_t full_size;               // Size of full data set (in bytes, multiple of mix size (128)).
	uint64_t cache_size;              // Size of compute cache (in bytes, multiple of node size (64)).
} ethash_params;

typedef struct ethash_return_value {
	uint8_t result[32];
	uint8_t mix_hash[32];
} ethash_return_value;

uint64_t ethash_get_datasize(const uint32_t block_number);
uint64_t ethash_get_cachesize(const uint32_t block_number);

// initialize the parameters
static inline void ethash_params_init(ethash_params *params, const uint32_t block_number) {
	params->full_size = ethash_get_datasize(block_number);
	params->cache_size = ethash_get_cachesize(block_number);
}

/***********************************
 * OLD API *************************
 ***********************************
 ******************** (deprecated) *
 ***********************************/

void ethash_get_seedhash(uint8_t seedhash[32], const uint32_t block_number);
void ethash_mkcache(void *cache, ethash_params const *params, const uint8_t seed[32]);
void ethash_light(ethash_return_value *ret, void const *cache, ethash_params const *params, const uint8_t header_hash[32], const uint64_t nonce);
void ethash_compute_full_data(void *mem, ethash_params const *params, void const *cache);
void ethash_full(ethash_return_value *ret, void const *full_mem, ethash_params const *params, const uint8_t header_hash[32], const uint64_t nonce);

/***********************************
 * NEW API *************************
 ***********************************/

// TODO: compute params and seed in ethash_new_light; it should take only block_number
// TODO: store params in ethash_light_t/ethash_full_t to avoid having to repass into compute/new_full

typedef uint8_t const ethash_seedhash_t[32];

typedef void const* ethash_light_t;
static inline ethash_light_t ethash_new_light(ethash_params const* params, ethash_seedhash_t seed) {
	void* ret = malloc((size_t)params->cache_size);
	ethash_mkcache(ret, params, seed);
	return ret;
}
static inline void ethash_compute_light(ethash_return_value *ret, ethash_light_t light, ethash_params const *params, const uint8_t header_hash[32], const uint64_t nonce) {
	ethash_light(ret, light, params, header_hash, nonce);
}
static inline void ethash_delete_light(ethash_light_t light) {
	free((void*)light);
}

typedef void const* ethash_full_t;
static inline ethash_full_t ethash_new_full(ethash_params const* params, ethash_light_t light) {
	void* ret = malloc((size_t)params->full_size);
	ethash_compute_full_data(ret, params, light);
	return ret;
}
static inline void ethash_prep_full(void *full, ethash_params const *params, void const *cache) {
	ethash_compute_full_data(full, params, cache);
}
static inline void ethash_compute_full(ethash_return_value *ret, void const *full, ethash_params const *params, const uint8_t header_hash[32], const uint64_t nonce) {
	ethash_full(ret, full, params, header_hash, nonce);
}

/// @brief Compare two s256-bit big-endian values.
/// @returns 1 if @a a is less than or equal to @a b, 0 otherwise.
/// Both parameters are 256-bit big-endian values.
static inline int ethash_leq_be256(const uint8_t a[32], const uint8_t b[32]) {
	// Boundary is big endian
	for (int i = 0; i < 32; i++) {
		if (a[i] == b[i])
			continue;
		return a[i] < b[i];
	}
	return 1;
}

/// Perofrms a cursory check on the validity of the nonce.
/// @returns 1 if the nonce may possibly be valid for the given header_hash & boundary.
/// @p boundary equivalent to 2 ^ 256 / block_difficulty, represented as a 256-bit big-endian.
int ethash_preliminary_check_boundary(
	const uint8_t header_hash[32],
	const uint64_t nonce,
	const uint8_t mix_hash[32],
	const uint8_t boundary[32]);

#define ethash_quick_check_difficulty ethash_preliminary_check_boundary
#define ethash_check_difficulty ethash_leq_be256

#ifdef __cplusplus
}
#endif
