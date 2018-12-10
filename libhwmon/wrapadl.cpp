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
    {
        cwarn << "Failed to obtain all required ADL function pointers";
        cwarn << "AMD hardware monitoring disabled";
        return nullptr;
    }


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
        cwarn << "Failed to obtain all required ADL function pointers";
        cwarn << "AMD hardware monitoring disabled";

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
        if (res != WRAPADL_OK)
        {
            cwarn << "Failed to obtain using adlAdapterAdapterInfoGet().";
            cwarn << "AMD hardware monitoring disabled";

            wrap_dlclose(adlh->adl_dll);
            free(adlh);
            return nullptr;
        }

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
        return -1;

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
    
    if (gpuindex < 0 || gpuindex >= adlh->adl_gpucount)
        return -1;

    ADLTemperature* temperature = new ADLTemperature();

    if (adlh->adlOverdrive5TemperatureGet(adlh->phys_logi_device_id[gpuindex], 0, temperature) !=
        WRAPADL_OK)
        return -1;

    *tempC = unsigned(temperature->iTemperature / 1000);
    delete temperature;
    return 0;
}

int wrap_adl_get_fanpcnt(wrap_adl_handle* adlh, int gpuindex, unsigned int* fanpcnt)
{

    if (gpuindex < 0 || gpuindex >= adlh->adl_gpucount)
        return -1;

    ADLFanSpeedValue* fan = new ADLFanSpeedValue();
    fan->iSpeedType = 1;

    if (adlh->adlOverdrive5FanSpeedGet(adlh->phys_logi_device_id[gpuindex], 0, fan) != WRAPADL_OK)
        return -1;

    *fanpcnt = unsigned(fan->iFanSpeed);
    delete fan;
    return 0;
}

int wrap_adl_get_power_usage(wrap_adl_handle* adlh, int gpuindex, unsigned int* miliwatts)
{

    if (gpuindex < 0 || gpuindex >= adlh->adl_gpucount)
        return -1;

    int power = 0;
    if (adlh->adl2Overdrive6CurrentPowerGet(
            adlh->context, adlh->phys_logi_device_id[gpuindex], 0, &power) != WRAPADL_OK)
        return -1;

    *miliwatts = (unsigned int)(power * 3.90625);
    return 0;
}

#if defined(__cplusplus)
}
#endif
