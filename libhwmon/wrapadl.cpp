/*
* Wrapper for ADL, inspired by wrapnvml from John E. Stone
*
* By Philipp Andreas - github@smurfy.de
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wraphelper.h"
#include "wrapadl.h"

#if defined(__cplusplus)
extern "C" {
#endif

void* ADL_API_CALL ADL_Main_Memory_Alloc(int iSize)
{
	void* lpBuffer = malloc(iSize);
	return lpBuffer;
}

wrap_adl_handle * wrap_adl_create()
{
	wrap_adl_handle *adlh = NULL;

#if defined(_WIN64)
	/* 64-bit Windows */
#define  libatiadlxx "atiadlxx.dll"
#elif defined(_WIN32) || defined(_MSC_VER)
	/* 32-bit Windows */
#define  libatiadlxx "atiadlxx.dll"
#elif defined(__linux) && (defined(__i386__) || defined(__ARM_ARCH_7A__))
	/* 32-bit linux assumed */
#define  libatiadlxx "libatiadlxx.so"
#elif defined(__linux)
	/* 64-bit linux assumed */
#define  libatiadlxx "libatiadlxx.so"
#else
#define  libatiadlxx ""
#warning "Unrecognized platform: need ADL DLL path for this platform..."
return NULL;
#endif

#if WIN32
	char tmp[512];
	ExpandEnvironmentStringsA(libatiadlxx, tmp, sizeof(tmp));
#else
	char tmp[512] = libatiadlxx;
#endif

	void *adl_dll = wrap_dlopen(tmp);
	if (adl_dll == NULL)
		return NULL;

	adlh = (wrap_adl_handle *)calloc(1, sizeof(wrap_adl_handle));

	adlh->adl_dll = adl_dll;

	adlh->adlMainControlCreate = (wrap_adlReturn_t(*)(ADL_MAIN_MALLOC_CALLBACK, int))
		wrap_dlsym(adlh->adl_dll, "ADL_Main_Control_Create");
	adlh->adlAdapterNumberOfAdapters = (wrap_adlReturn_t(*)(int *))
		wrap_dlsym(adlh->adl_dll, "ADL_Adapter_NumberOfAdapters_Get");
	adlh->adlAdapterAdapterInfoGet = (wrap_adlReturn_t(*)(LPAdapterInfo, int))
		wrap_dlsym(adlh->adl_dll, "ADL_Adapter_AdapterInfo_Get");
	adlh->adlAdapterAdapterIdGet = (wrap_adlReturn_t(*)(int, int*))
		wrap_dlsym(adlh->adl_dll, "ADL_Adapter_ID_Get");
	adlh->adlOverdrive5TemperatureGet = (wrap_adlReturn_t(*)(int, int, ADLTemperature*))
		wrap_dlsym(adlh->adl_dll, "ADL_Overdrive5_Temperature_Get");
	adlh->adlOverdrive5FanSpeedGet = (wrap_adlReturn_t(*)(int, int, ADLFanSpeedValue*))
		wrap_dlsym(adlh->adl_dll, "ADL_Overdrive5_FanSpeed_Get");
	adlh->adlMainControlRefresh = (wrap_adlReturn_t(*)(void))
		wrap_dlsym(adlh->adl_dll, "ADL_Main_Control_Refresh");
	adlh->adlMainControlDestory = (wrap_adlReturn_t(*)(void))
		wrap_dlsym(adlh->adl_dll, "ADL_Main_Control_Destroy");

	if (adlh->adlMainControlCreate == NULL ||
		adlh->adlMainControlDestory == NULL ||
		adlh->adlMainControlRefresh == NULL ||
		adlh->adlAdapterNumberOfAdapters == NULL ||
		adlh->adlAdapterAdapterInfoGet == NULL ||
		adlh->adlAdapterAdapterIdGet == NULL ||
		adlh->adlOverdrive5TemperatureGet == NULL ||
		adlh->adlOverdrive5FanSpeedGet == NULL
		) {
#if 0
		printf("Failed to obtain all required ADL function pointers\n");
#endif
		wrap_dlclose(adlh->adl_dll);
		free(adlh);
		return NULL;
	}

	adlh->adlMainControlCreate(ADL_Main_Memory_Alloc, 1); 
	adlh->adlMainControlRefresh();

	int logicalGpuCount = 0;
	adlh->adlAdapterNumberOfAdapters(&logicalGpuCount);

	adlh->phys_logi_device_id = (int*)calloc(logicalGpuCount, sizeof(int));
	
	adlh->adl_gpucount = 0;
	int last_adapter = 0;
	if (logicalGpuCount > 0) {
		adlh->devs = (LPAdapterInfo)malloc(sizeof(AdapterInfo) * logicalGpuCount);
		memset(adlh->devs, '\0', sizeof(AdapterInfo) * logicalGpuCount);

		adlh->devs->iSize = sizeof(adlh->devs);

		int res = adlh->adlAdapterAdapterInfoGet(adlh->devs, sizeof(AdapterInfo) * logicalGpuCount);

		for (int i = 0; i < logicalGpuCount; i++) {
			int adapterIndex = adlh->devs[i].iAdapterIndex;
			int adapterID = 0;

			res = adlh->adlAdapterAdapterIdGet(adapterIndex, &adapterID);

			if (res != WRAPADL_OK) {
				continue;
			}

			adlh->phys_logi_device_id[adlh->adl_gpucount] = adapterIndex;

			if (adapterID == last_adapter) {
				continue;
			}
			last_adapter = adapterID;
			adlh->adl_gpucount++;
		}
	}

	return adlh;
}

int wrap_adl_destory(wrap_adl_handle *adlh)
{
	adlh->adlMainControlDestory();
	wrap_dlclose(adlh->adl_dll);
	free(adlh);
	return 0;
}

int wrap_adl_get_gpucount(wrap_adl_handle *adlh, int *gpucount)
{
	*gpucount = adlh->adl_gpucount;
	return 0;
}

int wrap_adl_get_gpu_name(wrap_adl_handle *adlh, int gpuindex, char *namebuf, int bufsize)
{
	if (gpuindex < 0 || gpuindex >= adlh->adl_gpucount)
		return -1;

	memcpy(namebuf, adlh->devs[adlh->phys_logi_device_id[gpuindex]].strAdapterName, bufsize);
	return 0;
}


int wrap_adl_get_tempC(wrap_adl_handle *adlh, int gpuindex, unsigned int *tempC)
{
	wrap_adlReturn_t rc;
	if (gpuindex < 0 || gpuindex >= adlh->adl_gpucount)
		return -1;

	ADLTemperature *temperature = new ADLTemperature();
	rc = adlh->adlOverdrive5TemperatureGet(adlh->phys_logi_device_id[gpuindex], 0, temperature);
	if (rc != WRAPADL_OK) {
		return -1;
	}
	*tempC = unsigned(temperature->iTemperature / 1000);
	free(temperature);
	return 0;
}

int wrap_adl_get_fanpcnt(wrap_adl_handle *adlh, int gpuindex, unsigned int *fanpcnt)
{
	wrap_adlReturn_t rc;
	if (gpuindex < 0 || gpuindex >= adlh->adl_gpucount)
		return -1;

	ADLFanSpeedValue *fan = new ADLFanSpeedValue();
	fan->iSpeedType = 1;
	rc = adlh->adlOverdrive5FanSpeedGet(adlh->phys_logi_device_id[gpuindex], 0, fan);
	if (rc != WRAPADL_OK) {
		return -1;
	}
	*fanpcnt = unsigned(fan->iFanSpeed);
	free(fan);
	return 0;
}

#if defined(__cplusplus)
}
#endif
