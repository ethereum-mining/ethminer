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
/** @file io.h
 * @author Lefteris Karapetsas <lefteris@ethdev.com>
 * @date 2015
 */
#pragma once
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include "endian.h"
#include "ethash.h"

#ifdef __cplusplus
extern "C" {
#endif
// Maximum size for mutable part of DAG file name
// 10 is for maximum number of digits of a uint32_t (for REVISION)
// 1 is for _ and 16 is for the first 16 hex digits for first 8 bytes of
// the seedhash and last 1 is for the null terminating character
// Reference: https://github.com/ethereum/wiki/wiki/Ethash-DAG
#define DAG_MUTABLE_NAME_MAX_SIZE (10 + 1 + 16 + 1)
/// Possible return values of @see ethash_io_prepare
enum ethash_io_rc {
	ETHASH_IO_FAIL = 0,      ///< There has been an IO failure
	ETHASH_IO_MEMO_MISMATCH, ///< The DAG file did not exist or there was revision/hash mismatch
	ETHASH_IO_MEMO_MATCH,    ///< DAG file existed and revision/hash matched. No need to do anything
};

/**
 * Prepares io for ethash
 *
 * Create the DAG directory and the DAG file if they don't exist.
 *
 * @param[in] dirname    A null terminated c-string of the path of the ethash
 *                       data directory. If it does not exist it's created.
 * @param[in] seedhash   The seedhash of the current block number, used in the
 *                       naming of the file as can be seen from the spec at:
 *                       https://github.com/ethereum/wiki/wiki/Ethash-DAG
 * @param[out] f         If the hash/revision combo matched then this will point
 *                       to an opened file handler for that file. User will then
 *                       have to close it.
 * @return               For possible return values @see enum ethash_io_rc
 */
enum ethash_io_rc ethash_io_prepare(char const *dirname, ethash_h256_t seedhash, FILE **f);

/**
 * An fopen wrapper for no-warnings crossplatform fopen.
 *
 * Msvc compiler considers fopen to be insecure and suggests to use their
 * alternative. This is a wrapper for this alternative. Another way is to
 * #define _CRT_SECURE_NO_WARNINGS, but disabling all security warnings does
 * not sound like a good idea.
 *
 * @param file_name        The path to the file to open
 * @param mode             Opening mode. Check fopen()
 * @return                 The FILE* or NULL in failure
 */
FILE *ethash_fopen(const char *file_name, const char *mode);
/**
 * An strncat wrapper for no-warnings crossplatform strncat.
 *
 * Msvc compiler considers strncat to be insecure and suggests to use their
 * alternative. This is a wrapper for this alternative. Another way is to
 * #define _CRT_SECURE_NO_WARNINGS, but disabling all security warnings does
 * not sound like a good idea.
 *
 * @param des              Destination buffer
 * @param dest_size        Maximum size of the destination buffer. This is the
 *                         extra argument for the MSVC secure strncat
 * @param src              Souce buffer
 * @param count            Number of bytes to copy from source
 * @return                 If all is well returns the dest buffer. If there is an
 *                         error returns NULL
 */
char *ethash_strncat(char *dest, size_t dest_size, const char *src, size_t count);

static inline bool ethash_io_mutable_name(uint32_t revision,
                                          ethash_h256_t *seed_hash,
                                          char *output)
{
    uint64_t hash = *((uint64_t*)seed_hash);
#if LITTLE_ENDIAN == BYTE_ORDER
    hash = ethash_swap_u64(hash);
#endif
    return snprintf(output, DAG_MUTABLE_NAME_MAX_SIZE, "%u_%016lx", revision, hash) >= 0;
}

static inline char *ethash_io_create_filename(char const *dirname,
											  char const* filename,
											  size_t filename_length)
{
	size_t dirlen = strlen(dirname);
	// in C the cast is not needed, but a C++ compiler will complain for invalid conversion
	char *name = (char*)malloc(dirlen + filename_length + 1);
	if (!name) {
		return NULL;
	}

	name[0] = '\0';
	ethash_strncat(name, dirlen + filename_length + 1, dirname, dirlen);
	ethash_strncat(name, dirlen + filename_length + 1, filename, filename_length);
	return name;
}


#ifdef __cplusplus
}
#endif
