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
#include "ethash.h"

#ifdef __cplusplus
extern "C" {
#endif

static const char DAG_FILE_NAME[] = "full";
static const char DAG_MEMO_NAME[] = "full.info";
// MSVC thinks that "static const unsigned int" is not a compile time variable. Sorry for the #define :(
#define DAG_MEMO_BYTESIZE 36

/// Possible return values of @see ethash_io_prepare
enum ethash_io_rc {
    ETHASH_IO_FAIL = 0,      ///< There has been an IO failure
    ETHASH_IO_MEMO_MISMATCH, ///< Memo file either did not exist or there was content mismatch
    ETHASH_IO_MEMO_MATCH,    ///< Memo file existed and contents matched. No need to do anything
};

/**
 * Prepares io for ethash
 *
 * Create the DAG directory if it does not exist, and check if the memo file matches.
 * If it does not match then it's deleted to pave the way for @ref ethash_io_write()
 *
 * @param dirname        A null terminated c-string of the path of the ethash
 *                       data directory. If it does not exist it's created.
 * @param seedhash       The seedhash of the current block number
 * @return               For possible return values @see enum ethash_io_rc
 */
enum ethash_io_rc ethash_io_prepare(char const *dirname, ethash_h256_t seedhash);

/**
 * Fully computes data and writes it to the file on disk.
 *
 * This function should be called after @see ethash_io_prepare() and only if
 * its return value is @c ETHASH_IO_MEMO_MISMATCH. Will write both the full data
 * and the memo file.
 *
 * @param[in] dirname        A null terminated c-string of the path of the ethash
 *                           data directory. Has to exist.
 * @param[in] params         An ethash_params object containing the full size
 *                           and the cache size
 * @param[in] seedhash       The seedhash of the current block number
 * @param[in] cache          The cache data. Would have usually been calulated by
 *                           @see ethash_prep_light().
 * @param[out] data          Pass a pointer to uint8_t by reference here. If the
 *                           function is succesfull then this point to the allocated
 *                           data calculated by @see ethash_prep_full(). Memory
 *                           ownership is transfered to the callee. Remember that
 *                           you eventually need to free this with a call to free().
 * @param[out] data_size     Pass a uint64_t by value. If the function is succesfull
 *                           then this will contain the number of bytes allocated
 *                           for @a data.
 * @return                   True for success and false in case of failure.
 */
bool ethash_io_write(char const *dirname,
                     ethash_params const* params,
                     ethash_h256_t seedhash,
                     void const* cache,
                     uint8_t **data,
                     uint64_t *data_size);

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
 * An stncat wrapper for no-warnings crossplatform strncat.
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

static inline void ethash_io_serialize_info(uint32_t revision,
                                            ethash_h256_t seed_hash,
                                            char *output)
{
    // if .info is only consumed locally we don't really care about endianess
    memcpy(output, &revision, 4);
    memcpy(output + 4, &seed_hash, 32);
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
