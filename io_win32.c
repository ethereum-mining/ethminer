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
/** @file io_win32.c
 * @author Lefteris Karapetsas <lefteris@ethdev.com>
 * @date 2015
 */

#include "io.h"
#include <direct.h>
#include <errno.h>
#include <stdio.h>

FILE *ethash_fopen(const char *file_name, const char *mode)
{
	FILE *f;
	return fopen_s(&f, file_name, mode) == 0 ? f : NULL;
}

char *ethash_strncat(char *dest, size_t dest_size, const char *src, size_t count)
{
	return strncat_s(dest, dest_size, src, count) == 0 ? dest : NULL;
}

enum ethash_io_rc ethash_io_prepare(char const *dirname, ethash_h256_t seedhash)
{
	char mutable_name[DAG_MUTABLE_NAME_MAX_SIZE];
	enum ethash_io_rc ret = ETHASH_IO_FAIL;

	// assert directory exists
	int rc = _mkdir(dirname);
	if (rc == -1 && errno != EEXIST) {
		goto end;
	}

	ethash_io_mutable_name(REVISION, &seedhash, mutable_name);
	char *tmpfile = ethash_io_create_filename(dirname, mutable_name, strlen(mutable_name));
	if (!tmpfile) {
		goto end;
	}

	// try to open the file
	FILE *f = ethash_fopen(tmpfile, "rb");
	if (!f) {
		// file does not exist, will need to be created
		ret = ETHASH_IO_MEMO_MISMATCH;
		goto free_memo;
	}

	ret = ETHASH_IO_MEMO_MATCH;
	*output_file = f;
free_memo:
	free(tmpfile);
end:
	return ret;
}
