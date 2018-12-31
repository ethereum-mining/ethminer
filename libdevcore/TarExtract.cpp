/*
    This file is part of ethminer.

    ethminer is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ethminer is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ethminer.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <fstream>

#include <string.h>

#include <zlib.h>

#include "TarExtract.h"

#define min(a, b) (((a) < (b)) ? (a) : (b))

using namespace std;

#define BLK_SZ 512

// Decode a TAR octal number.

static uint64_t decodeTarOctal(char* data, size_t size)
{
    uint64_t sum = 0;
    char* currentPtr = data;
    while ((*currentPtr != 0) && (*currentPtr != ' ') && (size_t(currentPtr - data) < size))
        sum = (sum * 8) + (*currentPtr++ - '0');
    return sum;
}

struct TARFileHeader
{
    char filename[100];  // NUL-terminated
    char mode[8];
    char uid[8];
    char gid[8];
    char fileSize[12];
    char lastModification[12];
    char checksum[8];
    char typeFlag;  // Also called link indicator for none-UStar format
    char linkedFileName[100];
    char ustarVersion[2];  // 00
    char ownerUserName[32];
    char ownerGroupName[32];
    char deviceMajorNumber[8];
    char deviceMinorNumber[8];
    char filenamePrefix[155];
    char padding[12];  // Nothing of interest, but relevant for checksum

    size_t getFileSize() { return decodeTarOctal(fileSize, sizeof(fileSize)); }
};


bool ExtractFromTar(const string& name, const string& member, std::vector<unsigned char>& bin)
{
    gzFile f = gzopen(name.c_str(), "rb");
    if (f == NULL)
        return false;

    // Initialize a zero-filled block we can compare against (zero-filled header block --> end of
    // TAR archive)
    char zeroBlock[BLK_SZ];
    char padBuffer[BLK_SZ];
    bool rc = false;
    char* fileData = nullptr;

    memset(zeroBlock, 0, BLK_SZ);

    // Start reading
    while (true)
    {  // Stop if end of file has been reached or any error occured

        TARFileHeader currentFileHeader;

        // Read the file header.
        if (gzread(f, &currentFileHeader, BLK_SZ) != BLK_SZ)
            break;

        // When a block with zeroes-only is found, the TAR archive ends here
        if (memcmp(&currentFileHeader, zeroBlock, BLK_SZ) == 0)
            break;

        string filename(
            currentFileHeader.filename, min((size_t)100, strlen(currentFileHeader.filename)));
        if (currentFileHeader.typeFlag != '0' && currentFileHeader.typeFlag != 0)
            break;

        // Set the filename from the current header
        filename = string(currentFileHeader.filename);
        // Now the metadata in the current file header is valie -- we can read the values.
        size_t size = currentFileHeader.getFileSize();
        // Read the file into memory
        fileData = new char[size];
        if (size_t(gzread(f, fileData, size)) != size)
            break;

        if (filename == member)
        {
            for (unsigned int i = 0; i < size; i++)
                bin.push_back(fileData[i]);
            rc = true;
            break;
        }
        delete[] fileData;
        fileData = nullptr;
        // In the tar archive, entire 512-byte-blocks are used for each file
        // Therefore we now have to skip the padded bytes.
        size_t paddingBytes =
            (BLK_SZ - (size % BLK_SZ));  // How long the padding to 512 bytes needs to be
        if (paddingBytes != BLK_SZ)
        {
            // Simply ignore the padding
            if (size_t(gzread(f, padBuffer, paddingBytes)) != paddingBytes)
                break;
        }
    }
    if (fileData != nullptr)
        delete[] fileData;
    gzclose(f);

    return rc;
}
