/*
 * Wrapper for AMD SysFS on linux, using adapted code from amdcovc by matszpk
 *
 * By Philipp Andreas - github@smurfy.de
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#if defined(__linux)
#include <dirent.h>
#endif

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <string>

#include "wrapamdsysfs.h"
#include "wraphelper.h"

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

wrap_amdsysfs_handle* wrap_amdsysfs_create()
{
    wrap_amdsysfs_handle* sysfsh = nullptr;

#if defined(__linux)
    namespace fs = boost::filesystem;
    std::vector<pciInfo> devices;  // Used to collect devices

    char dbuf[120];
    // Check directory exist
    fs::path drm_dir("/sys/class/drm");
    if (!fs::exists(drm_dir) || !fs::is_directory(drm_dir))
        return nullptr;

    // Regex patterns to identify directory elements
    std::regex cardPattern("^card[0-9]{1,}$");
    std::regex hwmonPattern("^hwmon[0-9]{1,}$");

    // Loop directory contents
    for (fs::directory_iterator dirEnt(drm_dir); dirEnt != fs::directory_iterator(); ++dirEnt)
    {
        // Skip non relevant entries
        if (!fs::is_directory(dirEnt->path()) ||
            !std::regex_match(dirEnt->path().filename().string(), cardPattern))
            continue;

        std::string devName = dirEnt->path().filename().string();
        unsigned int devIndex = std::stoi(devName.substr(4), nullptr, 10);
        unsigned int vendorId = 0;
        unsigned int hwmonIndex = UINT_MAX;

        // Get AMD cards only (vendor 4098)
        fs::path vendor_file("/sys/class/drm/" + devName + "/device/vendor");
        snprintf(dbuf, 120, "/sys/class/drm/%s/device/vendor", devName.c_str());
        if (!fs::exists(vendor_file) || !fs::is_regular_file(vendor_file) ||
            !getFileContentValue(dbuf, vendorId) || vendorId != 4098)
            continue;

        // Check it has dependant hwmon directory
        fs::path hwmon_dir("/sys/class/drm/" + devName + "/device/hwmon");
        if (!fs::exists(hwmon_dir) || !fs::is_directory(hwmon_dir))
            continue;

        // Loop subelements in hwmon directory
        for (fs::directory_iterator hwmonEnt(hwmon_dir); hwmonEnt != fs::directory_iterator();
             ++hwmonEnt)
        {
            // Skip non relevant entries
            if (!fs::is_directory(hwmonEnt->path()) ||
                !std::regex_match(hwmonEnt->path().filename().string(), hwmonPattern))
                continue;

            unsigned int v = std::stoi(hwmonEnt->path().filename().string().substr(5), nullptr, 10);
            hwmonIndex = std::min(hwmonIndex, v);
        }
        if (hwmonIndex == UINT_MAX)
            continue;

        // Detect Pci Id
        fs::path uevent_file("/sys/class/drm/" + devName + "/device/uevent");
        if (!fs::exists(uevent_file) || !fs::is_regular_file(uevent_file))
            continue;

        snprintf(dbuf, 120, "/sys/class/drm/card%d/device/uevent", devIndex);
        std::ifstream ifs(dbuf, std::ios::binary);
        std::string line;
        int PciDomain = -1, PciBus = -1, PciDevice = -1, PciFunction = -1;
        while (std::getline(ifs, line))
        {
            if (line.length() > 24 && line.substr(0, 13) == "PCI_SLOT_NAME")
            {
                std::string pciId = line.substr(14);
                std::vector<std::string> pciIdParts;
                boost::split(pciIdParts, pciId, [](char c) { return (c == ':' || c == '.'); });

                try
                {
                    PciDomain = std::stoi(pciIdParts.at(0), nullptr, 16);
                    PciBus = std::stoi(pciIdParts.at(1), nullptr, 16);
                    PciDevice = std::stoi(pciIdParts.at(2), nullptr, 16);
                    PciFunction = std::stoi(pciIdParts.at(3), nullptr, 16);
                }
                catch (const std::exception&)
                {
                    PciDomain = PciBus = PciDevice = PciFunction = -1;
                }
                break;
            }
        }

        // If we got an error skip
        if (PciDomain == -1)
            continue;

        // We got all information needed
        // push in the list of collected devices
        pciInfo pInfo = pciInfo();
        pInfo.DeviceId = devIndex;
        pInfo.HwMonId = hwmonIndex;
        pInfo.PciDomain = PciDomain;
        pInfo.PciBus = PciBus;
        pInfo.PciDevice = PciDevice;
        devices.push_back(pInfo);
    }

    // Nothing collected - exit
    if (!devices.size())
    {
        cwarn << "Failed to obtain all required AMD file pointers";
        cwarn << "AMD hardware monitoring disabled";
        return nullptr;
    }

    unsigned int gpucount = devices.size();
    sysfsh = (wrap_amdsysfs_handle*)calloc(1, sizeof(wrap_amdsysfs_handle));
    if (sysfsh == nullptr)
    {
        cwarn << "Failed allocate memory";
        cwarn << "AMD hardware monitoring disabled";
        return sysfsh;
    }
    sysfsh->sysfs_gpucount = gpucount;
    sysfsh->sysfs_device_id = (unsigned int*)calloc(gpucount, sizeof(unsigned int));
    sysfsh->sysfs_hwmon_id = (unsigned int*)calloc(gpucount, sizeof(unsigned int));
    sysfsh->sysfs_pci_domain_id = (unsigned int*)calloc(gpucount, sizeof(unsigned int));
    sysfsh->sysfs_pci_bus_id = (unsigned int*)calloc(gpucount, sizeof(unsigned int));
    sysfsh->sysfs_pci_device_id = (unsigned int*)calloc(gpucount, sizeof(unsigned int));

    gpucount = 0;
    for (auto const& device : devices)
    {
        sysfsh->sysfs_device_id[gpucount] = device.DeviceId;
        sysfsh->sysfs_hwmon_id[gpucount] = device.HwMonId;
        sysfsh->sysfs_pci_domain_id[gpucount] = device.PciDomain;
        sysfsh->sysfs_pci_bus_id[gpucount] = device.PciBus;
        sysfsh->sysfs_pci_device_id[gpucount] = device.PciDevice;
        gpucount++;
    }

#endif
    return sysfsh;
}

int wrap_amdsysfs_destroy(wrap_amdsysfs_handle* sysfsh)
{
    free(sysfsh);
    return 0;
}

int wrap_amdsysfs_get_gpucount(wrap_amdsysfs_handle* sysfsh, int* gpucount)
{
    *gpucount = sysfsh->sysfs_gpucount;
    return 0;
}

int wrap_amdsysfs_get_tempC(wrap_amdsysfs_handle* sysfsh, int index, unsigned int* tempC)
{
    if (index < 0 || index >= sysfsh->sysfs_gpucount)
        return -1;

    int gpuindex = sysfsh->sysfs_device_id[index];

    int hwmonindex = sysfsh->sysfs_hwmon_id[index];
    if (hwmonindex < 0)
        return -1;

    char dbuf[120];
    snprintf(
        dbuf, 120, "/sys/class/drm/card%d/device/hwmon/hwmon%d/temp1_input", gpuindex, hwmonindex);

    unsigned int temp = 0;
    getFileContentValue(dbuf, temp);

    if (temp > 0)
        *tempC = temp / 1000;

    return 0;
}

int wrap_amdsysfs_get_fanpcnt(wrap_amdsysfs_handle* sysfsh, int index, unsigned int* fanpcnt)
{
    if (index < 0 || index >= sysfsh->sysfs_gpucount)
        return -1;

    int gpuindex = sysfsh->sysfs_device_id[index];
    int hwmonindex = sysfsh->sysfs_hwmon_id[index];
    if (hwmonindex < 0)
        return -1;

    unsigned int pwm = 0, pwmMax = 255, pwmMin = 0;

    char dbuf[120];
    snprintf(dbuf, 120, "/sys/class/drm/card%d/device/hwmon/hwmon%d/pwm1", gpuindex, hwmonindex);
    getFileContentValue(dbuf, pwm);

    snprintf(
        dbuf, 120, "/sys/class/drm/card%d/device/hwmon/hwmon%d/pwm1_max", gpuindex, hwmonindex);
    getFileContentValue(dbuf, pwmMax);

    snprintf(
        dbuf, 120, "/sys/class/drm/card%d/device/hwmon/hwmon%d/pwm1_min", gpuindex, hwmonindex);
    getFileContentValue(dbuf, pwmMin);

    *fanpcnt = (unsigned int)(double(pwm - pwmMin) / double(pwmMax - pwmMin) * 100.0);
    return 0;
}

int wrap_amdsysfs_get_power_usage(wrap_amdsysfs_handle* sysfsh, int index, unsigned int* milliwatts)
{
    try
    {
        if (index < 0 || index >= sysfsh->sysfs_gpucount)
            return -1;

        int gpuindex = sysfsh->sysfs_device_id[index];

        char dbuf[120];
        snprintf(dbuf, 120, "/sys/kernel/debug/dri/%d/amdgpu_pm_info", gpuindex);

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
