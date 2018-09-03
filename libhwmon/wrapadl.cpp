/*
 * Wrapper for ADL, inspired by wrapnvml from John E. Stone
 *
 * By Philipp Andreas - github@smurfy.de
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include "wrapadl.h"
#include "wraphelper.h"

#if defined(__cplusplus)
extern "C" {
#endif

void* ADL_API_CALL ADL_Main_Memory_Alloc(int iSize)
{
    void* lpBuffer = malloc(iSize);
    return lpBuffer;
}

wrap_adl_handle* wrap_adl_create()
{
    wrap_adl_handle* adlh = nullptr;

#if defined(_WIN32)
    /* Windows */
#define libatiadlxx "atiadlxx.dll"
#elif defined(__linux) && (defined(__i386__) || defined(__ARM_ARCH_7A__))
    /* 32-bit linux assumed */
#define libatiadlxx "libatiadlxx.so"
#elif defined(__linux)
    /* 64-bit linux assumed */
#define libatiadlxx "libatiadlxx.so"
#else
#define libatiadlxx ""
#warning "Unrecognized platform: need ADL DLL path for this platform..."
    return nullptr;
#endif

#ifdef _WIN32
    char tmp[512];
    ExpandEnvironmentStringsA(libatiadlxx, tmp, sizeof(tmp));
#else
    char tmp[512] = libatiadlxx;
#endif

    void* adl_dll = wrap_dlopen(tmp);
    if (adl_dll == nullptr)
        return nullptr;

    adlh = (wrap_adl_handle*)calloc(1, sizeof(wrap_adl_handle));

    adlh->adl_dll = adl_dll;

    adlh->adlMainControlCreate = (wrap_adlReturn_t(*)(ADL_MAIN_MALLOC_CALLBACK, int))wrap_dlsym(
        adlh->adl_dll, "ADL_Main_Control_Create");
    adlh->adlAdapterNumberOfAdapters =
        (wrap_adlReturn_t(*)(int*))wrap_dlsym(adlh->adl_dll, "ADL_Adapter_NumberOfAdapters_Get");
    adlh->adlAdapterAdapterInfoGet = (wrap_adlReturn_t(*)(LPAdapterInfo, int))wrap_dlsym(
        adlh->adl_dll, "ADL_Adapter_AdapterInfo_Get");
    adlh->adlAdapterAdapterIdGet =
        (wrap_adlReturn_t(*)(int, int*))wrap_dlsym(adlh->adl_dll, "ADL_Adapter_ID_Get");
    adlh->adlOverdrive5TemperatureGet = (wrap_adlReturn_t(*)(int, int, ADLTemperature*))wrap_dlsym(
        adlh->adl_dll, "ADL_Overdrive5_Temperature_Get");
    adlh->adlOverdrive5FanSpeedGet = (wrap_adlReturn_t(*)(int, int, ADLFanSpeedValue*))wrap_dlsym(
        adlh->adl_dll, "ADL_Overdrive5_FanSpeed_Get");
    adlh->adlMainControlRefresh =
        (wrap_adlReturn_t(*)(void))wrap_dlsym(adlh->adl_dll, "ADL_Main_Control_Refresh");
    adlh->adlMainControlDestroy =
        (wrap_adlReturn_t(*)(void))wrap_dlsym(adlh->adl_dll, "ADL_Main_Control_Destroy");
    adlh->adl2MainControlCreate = (wrap_adlReturn_t(*)(ADL_MAIN_MALLOC_CALLBACK, int,
        ADL_CONTEXT_HANDLE*))wrap_dlsym(adlh->adl_dll, "ADL2_Main_Control_Create");
    adlh->adl2MainControlDestroy = (wrap_adlReturn_t(*)(ADL_CONTEXT_HANDLE))wrap_dlsym(
        adlh->adl_dll, "ADL_Main_Control_Destroy");
    adlh->adl2Overdrive6CurrentPowerGet = (wrap_adlReturn_t(*)(ADL_CONTEXT_HANDLE, int, int,
        int*))wrap_dlsym(adlh->adl_dll, "ADL2_Overdrive6_CurrentPower_Get");
    adlh->adl2MainControlRefresh = (wrap_adlReturn_t(*)(ADL_CONTEXT_HANDLE))wrap_dlsym(
        adlh->adl_dll, "ADL2_Main_Control_Refresh");


    if (adlh->adlMainControlCreate == nullptr || adlh->adlMainControlDestroy == nullptr ||
        adlh->adlMainControlRefresh == nullptr || adlh->adlAdapterNumberOfAdapters == nullptr ||
        adlh->adlAdapterAdapterInfoGet == nullptr || adlh->adlAdapterAdapterIdGet == nullptr ||
        adlh->adlOverdrive5TemperatureGet == nullptr || adlh->adlOverdrive5FanSpeedGet == nullptr ||
        adlh->adl2MainControlCreate == nullptr || adlh->adl2MainControlRefresh == nullptr ||
        adlh->adl2MainControlDestroy == nullptr || adlh->adl2Overdrive6CurrentPowerGet == nullptr)
    {
#if 0
        printf("Failed to obtain all required ADL function pointers\n");
#endif
        wrap_dlclose(adlh->adl_dll);
        free(adlh);
        return nullptr;
    }

    adlh->adlMainControlCreate(ADL_Main_Memory_Alloc, 1);
    adlh->adlMainControlRefresh();

    adlh->context = nullptr;

    adlh->adl2MainControlCreate(ADL_Main_Memory_Alloc, 1, &(adlh->context));
    adlh->adl2MainControlRefresh(adlh->context);

    int logicalGpuCount = 0;
    adlh->adlAdapterNumberOfAdapters(&logicalGpuCount);

    adlh->phys_logi_device_id = (int*)calloc(logicalGpuCount, sizeof(int));

    adlh->adl_gpucount = 0;
    int last_adapter = 0;
    if (logicalGpuCount > 0)
    {
        adlh->log_gpucount = logicalGpuCount;
        adlh->devs = (LPAdapterInfo)malloc(sizeof(AdapterInfo) * logicalGpuCount);
        memset(adlh->devs, '\0', sizeof(AdapterInfo) * logicalGpuCount);

        adlh->devs->iSize = sizeof(adlh->devs);

        int res = adlh->adlAdapterAdapterInfoGet(adlh->devs, sizeof(AdapterInfo) * logicalGpuCount);

        for (int i = 0; i < logicalGpuCount; i++)
        {
            int adapterIndex = adlh->devs[i].iAdapterIndex;
            int adapterID = 0;

            res = adlh->adlAdapterAdapterIdGet(adapterIndex, &adapterID);

            if (res != WRAPADL_OK)
            {
                continue;
            }

            adlh->phys_logi_device_id[adlh->adl_gpucount] = adapterIndex;

            if (adapterID == last_adapter)
            {
                continue;
            }

            last_adapter = adapterID;
            adlh->adl_gpucount++;
        }
    }


    adlh->adl_opencl_device_id = (int*)calloc(adlh->adl_gpucount, sizeof(int));
#if ETH_ETHASHCL
    if (adlh->adl_gpucount > 0)
    {
        // Get and count OpenCL devices.
        adlh->opencl_gpucount = 0;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        std::vector<cl::Device> platdevs;
        for (unsigned p = 0; p < platforms.size(); p++)
        {
            std::string platformName = platforms[p].getInfo<CL_PLATFORM_NAME>();
            if (platformName == "AMD Accelerated Parallel Processing")
            {
                platforms[p].getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, &platdevs);
                adlh->opencl_gpucount = platdevs.size();
                break;
            }
        }
        adlh->opencl_adl_device_id = (int*)calloc(adlh->opencl_gpucount, sizeof(int));

        // Map ADL phys device id to Opencl
        for (int i = 0; i < adlh->adl_gpucount; i++)
        {
            for (unsigned j = 0; j < platdevs.size(); j++)
            {
                cl::Device cldev = platdevs[j];
                cl_device_topology_amd topology;
                int status = clGetDeviceInfo(cldev(), CL_DEVICE_TOPOLOGY_AMD,
                    sizeof(cl_device_topology_amd), &topology, nullptr);
                if (status == CL_SUCCESS)
                {
                    if (topology.raw.type == CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD)
                    {
                        if (adlh->devs[adlh->phys_logi_device_id[i]].iBusNumber ==
                                (int)topology.pcie.bus &&
                            adlh->devs[adlh->phys_logi_device_id[i]].iDeviceNumber ==
                                (int)topology.pcie.device &&
                            adlh->devs[adlh->phys_logi_device_id[i]].iFunctionNumber ==
                                (int)topology.pcie.function)
                        {
#if 0
                            printf("[DEBUG] - ADL GPU[%d]%d,%d,%d matches OpenCL GPU[%d]%d,%d,%d\n",
                            adlh->phys_logi_device_id[i],
                            adlh->devs[adlh->phys_logi_device_id[i]].iBusNumber,
                            adlh->devs[adlh->phys_logi_device_id[i]].iDeviceNumber,
                            adlh->devs[adlh->phys_logi_device_id[i]].iFunctionNumber,
                            j, (int)topology.pcie.bus, (int)topology.pcie.device, (int)topology.pcie.function);
#endif
                            adlh->adl_opencl_device_id[i] = j;
                            adlh->opencl_adl_device_id[j] = i;
                        }
                    }
                }
            }
        }
    }
#endif

    return adlh;
}

int wrap_adl_destroy(wrap_adl_handle* adlh)
{
    adlh->adlMainControlDestroy();
    adlh->adl2MainControlDestroy(adlh->context);
    wrap_dlclose(adlh->adl_dll);
    free(adlh);
    return 0;
}

int wrap_adl_get_gpucount(wrap_adl_handle* adlh, int* gpucount)
{
    *gpucount = adlh->adl_gpucount;
    return 0;
}

int wrap_adl_get_gpu_name(wrap_adl_handle* adlh, int gpuindex, char* namebuf, int bufsize)
{
    if (gpuindex < 0 || gpuindex >= adlh->adl_gpucount)
        return -1;

    memcpy(namebuf, adlh->devs[adlh->phys_logi_device_id[gpuindex]].strAdapterName, bufsize);
    return 0;
}

int wrap_adl_get_gpu_pci_id(wrap_adl_handle* adlh, int gpuindex, char* idbuf, int bufsize)
{
    if (gpuindex < 0 || gpuindex >= adlh->adl_gpucount)
    {
        return -1;
    }
    char buf[256];
    sprintf(buf, "%04x:%02x:%02x",
        0,  // Is probably 0
        adlh->devs[adlh->phys_logi_device_id[gpuindex]].iBusNumber,
        adlh->devs[adlh->phys_logi_device_id[gpuindex]].iDeviceNumber);
    memcpy(idbuf, buf, bufsize);
    return 0;
}

int wrap_adl_get_tempC(wrap_adl_handle* adlh, int gpuindex, unsigned int* tempC)
{
    wrap_adlReturn_t rc;
    if (gpuindex < 0 || gpuindex >= adlh->adl_gpucount)
        return -1;

    ADLTemperature* temperature = new ADLTemperature();
    rc = adlh->adlOverdrive5TemperatureGet(adlh->phys_logi_device_id[gpuindex], 0, temperature);
    if (rc != WRAPADL_OK)
    {
        return -1;
    }
    *tempC = unsigned(temperature->iTemperature / 1000);
    delete temperature;
    return 0;
}

int wrap_adl_get_fanpcnt(wrap_adl_handle* adlh, int gpuindex, unsigned int* fanpcnt)
{
    wrap_adlReturn_t rc;
    if (gpuindex < 0 || gpuindex >= adlh->adl_gpucount)
        return -1;

    ADLFanSpeedValue* fan = new ADLFanSpeedValue();
    fan->iSpeedType = 1;
    rc = adlh->adlOverdrive5FanSpeedGet(adlh->phys_logi_device_id[gpuindex], 0, fan);
    if (rc != WRAPADL_OK)
    {
        return -1;
    }
    *fanpcnt = unsigned(fan->iFanSpeed);
    delete fan;
    return 0;
}

int wrap_adl_get_power_usage(wrap_adl_handle* adlh, int gpuindex, unsigned int* miliwatts)
{
    wrap_adlReturn_t rc;
    if (gpuindex < 0 || gpuindex >= adlh->adl_gpucount)
        return -1;

    int power = 0;
    rc = adlh->adl2Overdrive6CurrentPowerGet(
        adlh->context, adlh->phys_logi_device_id[gpuindex], 0, &power);
    *miliwatts = (unsigned int)(power * 3.90625);
    return rc;
}

#if defined(__cplusplus)
}
#endif
