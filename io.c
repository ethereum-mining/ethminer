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
/** @file io.c
 * @author Lefteris Karapetsas <lefteris@ethdev.com>
 * @date 2015
 */
#include "io.h"
#include <string.h>
#include <stdio.h>

static bool ethash_io_write_file(char const *dirname,
								 char const* filename,
								 size_t filename_length,
								 void const* data,
								 size_t data_size)
{
	bool ret = false;
	char *fullname = ethash_io_create_filename(dirname, filename, filename_length);
	if (!fullname) {
		return false;
	}
	FILE *f = ethash_fopen(fullname, "wb");
	if (!f) {
		goto free_name;
	}
	if (data_size != fwrite(data, 1, data_size, f)) {
		goto close;
	}

	ret = true;
close:
	fclose(f);
free_name:
	free(fullname);
	return ret;
}
