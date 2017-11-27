/*
* Wrapper for AMD SysFS on linux, using adapted code from amdcovc by matszpk
*
* By Philipp Andreas - github@smurfy.de
*/
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <cstring>
#include <climits>
#include <limits>
#include <cstdint>
#include <algorithm>
#include <fstream>
#include <sys/types.h>
#if defined(__linux)
#include <dirent.h>
#endif
#include "wrapamdsysfs.h"

static bool getFileContentValue(const char* filename, unsigned int& value)
{
	value = 0;
	std::ifstream ifs(filename, std::ios::binary);
	std::string line;
	std::getline(ifs, line);
	char* p = (char*)line.c_str();
	char* p2;
	errno = 0;
	value = strtoul(p, &p2, 0);
	if (errno != 0)
		return false;
	return (p != p2);
}


wrap_amdsysfs_handle * wrap_amdsysfs_create()
{
	wrap_amdsysfs_handle *sysfsh = NULL;

#if defined(__linux)
	sysfsh = (wrap_amdsysfs_handle *)calloc(1, sizeof(wrap_amdsysfs_handle));

	DIR* dirp = opendir("/sys/class/drm");
	if (dirp == nullptr)
		return NULL;

	unsigned int gpucount = 0;
	struct dirent* dire;
	while ((dire = readdir(dirp)) != nullptr)
	{
		if (::strncmp(dire->d_name, "card", 4) != 0)
			continue; // is not card directory
		const char* p;
		for (p = dire->d_name + 4; ::isdigit(*p); p++);
		if (*p != 0)
			continue; // is not card directory
		unsigned int v = ::strtoul(dire->d_name + 4, nullptr, 10);
		gpucount = std::max(gpucount, v + 1);
	}
	if (errno != 0)
	{
		closedir(dirp);
		return NULL;
	}
	closedir(dirp);

	sysfsh->card_sysfs_device_id = (int*)calloc(gpucount, sizeof(int));
	sysfsh->sysfs_hwmon_id = (int*)calloc(gpucount, sizeof(int));

	// filter AMD GPU cards and create mappings
	char dbuf[120];
	int cardIndex = 0;
	for (unsigned int i = 0; i < gpucount; i++)
	{
		sysfsh->card_sysfs_device_id[cardIndex] = -1;
		sysfsh->sysfs_hwmon_id[cardIndex] = -1;

		snprintf(dbuf, 120, "/sys/class/drm/card%u/device/vendor", i);
		unsigned int vendorId = 0;
		if (!getFileContentValue(dbuf, vendorId))
			continue;
		if (vendorId != 4098) // if not AMD
			continue;

		sysfsh->card_sysfs_device_id[cardIndex] = i;
		cardIndex++;
	}

	// Number of AMD cards found we do not care about non AMD cards
	sysfsh->sysfs_gpucount = cardIndex;

	// Get hwmon directory index
	for (int i = 0; i < sysfsh->sysfs_gpucount; i++)
	{
		int sysfsIdx = sysfsh->card_sysfs_device_id[i];

		// Should not happen
		if (sysfsIdx < 0) {
			free(sysfsh);
			return NULL;
		}

		// search hwmon
		errno = 0;
		snprintf(dbuf, 120, "/sys/class/drm/card%u/device/hwmon", sysfsIdx);
		DIR* dirp = opendir(dbuf);
		if (dirp == nullptr) {
			free(sysfsh);
			return NULL;
		}
		errno = 0;
		struct dirent* dire;
		unsigned int hwmonIndex = UINT_MAX;
		while ((dire = readdir(dirp)) != nullptr)
		{
			if (::strncmp(dire->d_name, "hwmon", 5) != 0)
				continue; // is not hwmon directory
			const char* p;
			for (p = dire->d_name + 5; ::isdigit(*p); p++);
			if (*p != 0)
				continue; // is not hwmon directory
			errno = 0;
			unsigned int v = ::strtoul(dire->d_name + 5, nullptr, 10);
			hwmonIndex = std::min(hwmonIndex, v);
		}
		if (errno != 0)
		{
			closedir(dirp);
			free(sysfsh);
			return NULL;
		}
		closedir(dirp);
		if (hwmonIndex == UINT_MAX) {
			free(sysfsh);
			return NULL;
		}

		sysfsh->sysfs_hwmon_id[i] = hwmonIndex;
	}
#endif

	return sysfsh;
}
int wrap_amdsysfs_destory(wrap_amdsysfs_handle *sysfsh)
{
	free(sysfsh);
	return 0;
}

int wrap_amdsysfs_get_gpucount(wrap_amdsysfs_handle *sysfsh, int *gpucount)
{
	*gpucount = sysfsh->sysfs_gpucount;
	return 0;
}

int wrap_amdsysfs_get_tempC(wrap_amdsysfs_handle *sysfsh, int index, unsigned int *tempC)
{
	int gpuindex = sysfsh->card_sysfs_device_id[index];
	if (gpuindex < 0 || index >= sysfsh->sysfs_gpucount)
		return -1;

	int hwmonindex = sysfsh->sysfs_hwmon_id[index];
	if (hwmonindex < 0)
		return -1;

	char dbuf[120];
	snprintf(dbuf, 120, "/sys/class/drm/card%u/device/hwmon/hwmon%u/temp1_input",
		gpuindex, hwmonindex);

	unsigned int temp = 0;
	getFileContentValue(dbuf, temp);

	if (temp > 0)
		*tempC = temp / 1000;

	return 0;
}

int wrap_amdsysfs_get_fanpcnt(wrap_amdsysfs_handle *sysfsh, int index, unsigned int *fanpcnt)
{
	int gpuindex = sysfsh->card_sysfs_device_id[index];
	if (gpuindex < 0 || index >= sysfsh->sysfs_gpucount)
		return -1;

	int hwmonindex = sysfsh->sysfs_hwmon_id[index];
	if (hwmonindex < 0)
		return -1;

	unsigned int pwm = 0, pwmMax = 255, pwmMin = 0;

	char dbuf[120];
	snprintf(dbuf, 120, "/sys/class/drm/card%u/device/hwmon/hwmon%u/pwm1",
		gpuindex, hwmonindex);
	getFileContentValue(dbuf, pwm);

	snprintf(dbuf, 120, "/sys/class/drm/card%u/device/hwmon/hwmon%u/pwm1_max",
		gpuindex, hwmonindex);
	getFileContentValue(dbuf, pwmMax);

	snprintf(dbuf, 120, "/sys/class/drm/card%u/device/hwmon/hwmon%u/pwm1_min",
		gpuindex, hwmonindex);
	getFileContentValue(dbuf, pwmMin);

	*fanpcnt = double(pwm - pwmMin) / double(pwmMax - pwmMin) * 100.0;
	return 0;
}