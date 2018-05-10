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
#include <regex>
#if defined(__linux)
#include <dirent.h>
#endif

#include <libdevcore/Log.h>

#include "wraphelper.h"
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
	errno = 0;
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

	sysfsh->opencl_gpucount = 0;
	sysfsh->sysfs_opencl_device_id = (int*)calloc(sysfsh->sysfs_gpucount, sizeof(int));
#if ETH_ETHASHCL
	if (sysfsh->sysfs_gpucount > 0) {
		//Get and count OpenCL devices.
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		std::vector<cl::Device> platdevs;
		for (unsigned p = 0; p < platforms.size(); p++) {
			std::string platformName = platforms[p].getInfo<CL_PLATFORM_NAME>();
			if (platformName == "AMD Accelerated Parallel Processing") {
				platforms[p].getDevices(
					CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR,
					&platdevs
				);
				sysfsh->opencl_gpucount = platdevs.size();
				break;
			}
		}
		sysfsh->opencl_sysfs_device_id = (int*)calloc(sysfsh->opencl_gpucount, sizeof(int));

		//Map SysFs to opencl devices
		for (int i = 0; i < sysfsh->sysfs_gpucount; i++) {
			for (unsigned j = 0; j < platdevs.size(); j++) {
				cl::Device cldev = platdevs[j];
				cl_device_topology_amd topology;
				int status = clGetDeviceInfo(cldev(), CL_DEVICE_TOPOLOGY_AMD,
					sizeof(cl_device_topology_amd), &topology, NULL);
				if (status == CL_SUCCESS) {
					if (topology.raw.type == CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD) {

						int gpuindex = sysfsh->card_sysfs_device_id[i];
						char dbuf[120];
						snprintf(dbuf, 120, "/sys/class/drm/card%u/device/uevent", gpuindex);
						std::ifstream ifs(dbuf, std::ios::binary);
						std::string line;
						int iBus = 0, iDevice = 0, iFunction = 0;
						while (std::getline(ifs, line)) {
							if (line.length() > 24 && line.substr(0, 13) == "PCI_SLOT_NAME") {
								std::string pciId = line.substr(14, 12);
								std::vector<std::string> pciParts;
								std::string part;
								std::size_t prev = 0, pos;
								while ((pos = pciId.find_first_of(":.", prev)) != std::string::npos)
								{
									if (pos > prev)
										pciParts.push_back(pciId.substr(prev, pos - prev));
									prev = pos + 1;
								}
								if (prev < pciId.length())
									pciParts.push_back(pciId.substr(prev, std::string::npos));

								//Format -- DDDD:BB:dd.FF 
								//??? Above comment doesn't match following statements!!!
								try {
									iBus = std::stoul(pciParts[1].c_str());
									iDevice = std::stoul(pciParts[2].c_str());
									iFunction = std::stoul(pciParts[3].c_str());
								}
								catch (...)
								{
									iBus = -1;
									iDevice = -1;
									iFunction = -1;
								}
								break;
							}
						}

						if (iBus == (int)topology.pcie.bus
							&& iDevice == (int)topology.pcie.device
							&& iFunction == (int)topology.pcie.function) {
#if 0
							printf("[DEBUG] - SYSFS GPU[%d]%d,%d,%d matches OpenCL GPU[%d]%d,%d,%d\n",
								i,
								iBus,
								iDevice,
								iFunction,
								j, (int)topology.pcie.bus, (int)topology.pcie.device, (int)topology.pcie.function);
#endif	
							sysfsh->sysfs_opencl_device_id[i] = j;
							sysfsh->opencl_sysfs_device_id[j] = i;
						}
					}
				}

			}
		}
	}
#endif

#endif
	return sysfsh;
}
int wrap_amdsysfs_destroy(wrap_amdsysfs_handle *sysfsh)
{
	free(sysfsh);
	return 0;
}

int wrap_amdsysfs_get_gpucount(wrap_amdsysfs_handle *sysfsh, int *gpucount)
{
	*gpucount = sysfsh->sysfs_gpucount;
	return 0;
}

int wrap_amdsysfs_get_gpu_pci_id(wrap_amdsysfs_handle *sysfsh, int index, char *idbuf, int bufsize)
{
	int gpuindex = sysfsh->card_sysfs_device_id[index];
	if (gpuindex < 0 || index >= sysfsh->sysfs_gpucount)
		return -1;

	char dbuf[120];
	snprintf(dbuf, 120, "/sys/class/drm/card%u/device/uevent", gpuindex);

	std::ifstream ifs(dbuf, std::ios::binary);
	std::string line;

	while (std::getline(ifs, line))
	{
		if (line.length() > 24 && line.substr(0, 13) == "PCI_SLOT_NAME") {
			memcpy(idbuf, line.substr(14, 12).c_str(), bufsize);
			return 0;
		}
	}

	//memcpy(idbuf, "0000:00:00.0", bufsize);//?
	return -1;
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

	*fanpcnt = (unsigned int)(double(pwm - pwmMin) / double(pwmMax - pwmMin) * 100.0);
	return 0;
}

int wrap_amdsysfs_get_power_usage(wrap_amdsysfs_handle* sysfsh, int index, unsigned int* milliwatts)
{
    try
    {
        int gpuindex = sysfsh->card_sysfs_device_id[index];
        if (gpuindex < 0 || index >= sysfsh->sysfs_gpucount)
            return -1;

        char dbuf[120];
        snprintf(dbuf, 120, "/sys/kernel/debug/dri/%u/amdgpu_pm_info", gpuindex);

        std::ifstream ifs(dbuf, std::ios::binary);
        std::string line;

        while (std::getline(ifs, line))
        {
            std::smatch sm;
            std::regex regex(R"(([\d|\.]+) W \(average GPU\))");
            if (std::regex_search(line, sm, regex))
            {
                if (sm.size() == 2)
                {
                    double watt = atof(sm.str(1).c_str());
                    *milliwatts = (unsigned int)(watt * 1000);
                    return 0;
                }
            }
        }
    }
    catch (const std::exception& ex)
    {
        cwarn << "Error in amdsysfs_get_power_usage: " << ex.what();
    }

    return -1;
}
