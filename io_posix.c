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
/** @file io_posix.c
 * @author Lefteris Karapetsas <lefteris@ethdev.com>
 * @date 2015
 */
#include "io.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <libgen.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

static const char DAG_FILE_NAME[] = "full";
static const char DAG_MEMO_NAME[] = "full.info";
static const unsigned int DAG_MEMO_BYTESIZE = 36;

bool ethash_io_prepare(char const *dirname, uint32_t block_number)
{
    char read_buffer[DAG_MEMO_BYTESIZE];
    char expect_buffer[DAG_MEMO_BYTESIZE];
    bool ret = false;

    // assert directory exists, full owner permissions and read/search for others
    int rc = mkdir(dirname, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (rc == -1 && errno != EEXIST) {
        goto end;
    }

    // try to open memo file
    char *memofile = malloc(strlen(dirname) + sizeof(DAG_MEMO_NAME));
    if (!memofile) {
        goto end;
    }

    FILE *f = fopen(memofile, "rb");
    if (!f) {
        // file does not exist, so no checking happens. All is fine.
        ret = true;
        goto free_memo;
    }

    if (fread(read_buffer, 1, DAG_MEMO_BYTESIZE, f) != DAG_MEMO_BYTESIZE) {
        goto free_memo;
    }

    ethash_io_serialize_info(REVISION, block_number, expect_buffer);
    if (memcmp(read_buffer, expect_buffer, DAG_MEMO_BYTESIZE) != 0) {
        // we have different memo contents so delete the memo file
        if (unlink(memofile) != 0) {
            goto free_memo;
        }
    }

    ret = true;

free_memo:
    free(memofile);
end:
    return ret;
}

void ethash_io_write()
{
}

