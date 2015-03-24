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
#include "ethash.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Prepares io for ethash
 * @param dirname        A null terminated c-string of the path of the ethash
 *                       data directory. If it does not exist it's created.
 * @param block_number   The current block number. Used in seedhash calculation.
 * @returns              True if all went fine, and false if there was any kind
 *                       of error
 */
bool ethash_io_prepare(char const *dirname, uint32_t block_number);
void ethash_io_write();
static inline void ethash_io_serialize_info(uint32_t revision,
                                            uint32_t block_number,
                                            char *output)
{
    // if .info is only consumed locally we don't really care about endianess
    memcpy(output, &revision, 4);
    ethash_get_seedhash((uint8_t*)(output + 4), block_number);
}


#ifdef __cplusplus
}
#endif
