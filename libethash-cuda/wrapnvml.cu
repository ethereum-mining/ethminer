/*
 * A trivial little dlopen()-based wrapper library for the
 * NVIDIA NVML library, to allow runtime discovery of NVML on an
 * arbitrary system.  This is all very hackish and simple-minded, but
 * it serves my immediate needs in the short term until NVIDIA provides
 * a static NVML wrapper library themselves, hopefully in
 * CUDA 6.5 or maybe sometime shortly after.
 *
 * This trivial code is made available under the "new" 3-clause BSD license,
 * and/or any of the GPL licenses you prefer.
 * Feel free to use the code and modify as you see fit.
 *
 * John E. Stone - john.stone@gmail.com
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "wrapnvml.h"
#include "cuda_runtime.h"

/*
 * Wrappers to emulate dlopen() on other systems like Windows
 */
#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
#include <windows.h>
static void *wrap_dlopen(const char *filename) {
  return (void *)LoadLibrary(filename);
}
static void *wrap_dlsym(void *h, const char *sym) {
  return (void *)GetProcAddress((HINSTANCE)h, sym);
}
static int wrap_dlclose(void *h) {
  /* FreeLibrary returns nonzero on success */
  return (!FreeLibrary((HINSTANCE)h));
}
#else
/* assume we can use dlopen itself... */
#include <dlfcn.h>
static void *wrap_dlopen(const char *filename) {
  return dlopen(filename, RTLD_NOW);
}
static void *wrap_dlsym(void *h, const char *sym) {
  return dlsym(h, sym);
}
static int wrap_dlclose(void *h) {
  return dlclose(h);
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif

wrap_nvml_handle * wrap_nvml_create() {
  int i=0;
  wrap_nvml_handle *nvmlh = NULL;

  /* 
   * We use hard-coded library installation locations for the time being...
   * No idea where or if libnvidia-ml.so is installed on MacOS X, a 
   * deep scouring of the filesystem on one of the Mac CUDA build boxes
   * I used turned up nothing, so for now it's not going to work on OSX.
   */
#if defined(_WIN64)
  /* 64-bit Windows */
#define  libnvidia_ml "%PROGRAMFILES%/NVIDIA Corporation/NVSMI/nvml.dll"
#elif defined(_WIN32) || defined(_MSC_VER)
  /* 32-bit Windows */
#define  libnvidia_ml "%PROGRAMFILES%/NVIDIA Corporation/NVSMI/nvml.dll"
#elif defined(__linux) && (defined(__i386__) || defined(__ARM_ARCH_7A__))
  /* 32-bit linux assumed */
#define  libnvidia_ml "/usr/lib32/libnvidia-ml.so"
#elif defined(__linux)
  /* 64-bit linux assumed */
#define  libnvidia_ml "/usr/lib/libnvidia-ml.so"
#else
#error "Unrecognized platform: need NVML DLL path for this platform..."
#endif

#if WIN32
  char tmp[512];
  ExpandEnvironmentStringsA(libnvidia_ml, tmp, sizeof(tmp)); 
#else
  char tmp[512] = libnvidia_ml;
#endif

  void *nvml_dll = wrap_dlopen(tmp);
  if (nvml_dll == NULL)
    return NULL;

  nvmlh = (wrap_nvml_handle *) calloc(1, sizeof(wrap_nvml_handle));

  nvmlh->nvml_dll = nvml_dll;  

  nvmlh->nvmlInit = (wrap_nvmlReturn_t (*)(void)) 
    wrap_dlsym(nvmlh->nvml_dll, "nvmlInit");
  nvmlh->nvmlDeviceGetCount = (wrap_nvmlReturn_t (*)(int *)) 
    wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetCount_v2");
  nvmlh->nvmlDeviceGetHandleByIndex = (wrap_nvmlReturn_t (*)(int, wrap_nvmlDevice_t *)) 
    wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetHandleByIndex_v2");
  nvmlh->nvmlDeviceGetPciInfo = (wrap_nvmlReturn_t (*)(wrap_nvmlDevice_t, wrap_nvmlPciInfo_t *)) 
    wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPciInfo");
  nvmlh->nvmlDeviceGetName = (wrap_nvmlReturn_t (*)(wrap_nvmlDevice_t, char *, int))
    wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetName");
  nvmlh->nvmlDeviceGetTemperature = (wrap_nvmlReturn_t (*)(wrap_nvmlDevice_t, int, unsigned int *))
    wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetTemperature");
  nvmlh->nvmlDeviceGetFanSpeed = (wrap_nvmlReturn_t (*)(wrap_nvmlDevice_t, unsigned int *))
    wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetFanSpeed");
  nvmlh->nvmlDeviceGetPowerUsage = (wrap_nvmlReturn_t (*)(wrap_nvmlDevice_t, unsigned int *))
    wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPowerUsage");
  nvmlh->nvmlShutdown = (wrap_nvmlReturn_t (*)()) 
    wrap_dlsym(nvmlh->nvml_dll, "nvmlShutdown");

  if (nvmlh->nvmlInit == NULL || 
      nvmlh->nvmlShutdown == NULL ||
      nvmlh->nvmlDeviceGetCount == NULL ||
      nvmlh->nvmlDeviceGetHandleByIndex == NULL || 
      nvmlh->nvmlDeviceGetPciInfo == NULL ||
      nvmlh->nvmlDeviceGetName == NULL ||
      nvmlh->nvmlDeviceGetTemperature == NULL ||
      nvmlh->nvmlDeviceGetFanSpeed == NULL ||
      nvmlh->nvmlDeviceGetPowerUsage == NULL
      ) {
#if 0
    printf("Failed to obtain all required NVML function pointers\n");
#endif
    wrap_dlclose(nvmlh->nvml_dll);
    free(nvmlh);
    return NULL;
  }

  nvmlh->nvmlInit();
  nvmlh->nvmlDeviceGetCount(&nvmlh->nvml_gpucount);

  /* Query CUDA device count, in case it doesn't agree with NVML, since  */
  /* CUDA will only report GPUs with compute capability greater than 1.0 */ 
  if (cudaGetDeviceCount(&nvmlh->cuda_gpucount) != cudaSuccess) {
#if 0
    printf("Failed to query CUDA device count!\n");
#endif
    wrap_dlclose(nvmlh->nvml_dll);
    free(nvmlh);
    return NULL;
  }

  nvmlh->devs = (wrap_nvmlDevice_t *) calloc(nvmlh->nvml_gpucount, sizeof(wrap_nvmlDevice_t));
  nvmlh->nvml_pci_domain_id = (unsigned int*) calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));
  nvmlh->nvml_pci_bus_id = (unsigned int*) calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));
  nvmlh->nvml_pci_device_id = (unsigned int*) calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));
  nvmlh->nvml_cuda_device_id = (int*) calloc(nvmlh->nvml_gpucount, sizeof(int));
  nvmlh->cuda_nvml_device_id = (int*) calloc(nvmlh->cuda_gpucount, sizeof(int));

  /* Obtain GPU device handles we're going to need repeatedly... */
  for (i=0; i<nvmlh->nvml_gpucount; i++) {
    nvmlh->nvmlDeviceGetHandleByIndex(i, &nvmlh->devs[i]);
  } 

  /* Query PCI info for each NVML device, and build table for mapping of */
  /* CUDA device IDs to NVML device IDs and vice versa                   */
  for (i=0; i<nvmlh->nvml_gpucount; i++) {
    wrap_nvmlPciInfo_t pciinfo;
    nvmlh->nvmlDeviceGetPciInfo(nvmlh->devs[i], &pciinfo);
    nvmlh->nvml_pci_domain_id[i] = pciinfo.domain;
    nvmlh->nvml_pci_bus_id[i]    = pciinfo.bus;
    nvmlh->nvml_pci_device_id[i] = pciinfo.device;
  }

  /* build mapping of NVML device IDs to CUDA IDs */
  for (i=0; i<nvmlh->nvml_gpucount; i++) {
    nvmlh->nvml_cuda_device_id[i] = -1;
  } 
  for (i=0; i<nvmlh->cuda_gpucount; i++) {
    cudaDeviceProp props;
    nvmlh->cuda_nvml_device_id[i] = -1;

    if (cudaGetDeviceProperties(&props, i) == cudaSuccess) {
      int j;
      for (j=0; j<nvmlh->nvml_gpucount; j++) {
        if ((nvmlh->nvml_pci_domain_id[j] == props.pciDomainID) &&
            (nvmlh->nvml_pci_bus_id[j]    == props.pciBusID) &&
            (nvmlh->nvml_pci_device_id[j] == props.pciDeviceID)) {
#if 0
          printf("CUDA GPU[%d] matches NVML GPU[%d]\n", i, j);
#endif
          nvmlh->nvml_cuda_device_id[j] = i;
          nvmlh->cuda_nvml_device_id[i] = j;
        }
      }
    }
  }

  return nvmlh;
}


int wrap_nvml_destroy(wrap_nvml_handle *nvmlh) {
  nvmlh->nvmlShutdown();

  wrap_dlclose(nvmlh->nvml_dll);
  free(nvmlh);
  return 0;
}


int wrap_nvml_get_gpucount(wrap_nvml_handle *nvmlh, int *gpucount) {
  *gpucount = nvmlh->nvml_gpucount;
  return 0; 
}

int wrap_cuda_get_gpucount(wrap_nvml_handle *nvmlh, int *gpucount) {
  *gpucount = nvmlh->cuda_gpucount;
  return 0; 
}

int wrap_nvml_get_gpu_name(wrap_nvml_handle *nvmlh,
                           int cudaindex, 
                           char *namebuf,
                           int bufsize) {
  int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
  if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
    return -1;

  if (nvmlh->nvmlDeviceGetName(nvmlh->devs[gpuindex], namebuf, bufsize) != WRAPNVML_SUCCESS)
    return -1; 

  return 0;
}


int wrap_nvml_get_tempC(wrap_nvml_handle *nvmlh,
                        int cudaindex, unsigned int *tempC) {
  wrap_nvmlReturn_t rc;
  int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
  if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
    return -1;

  rc = nvmlh->nvmlDeviceGetTemperature(nvmlh->devs[gpuindex], 0u /* NVML_TEMPERATURE_GPU */, tempC);
  if (rc != WRAPNVML_SUCCESS) {
    return -1; 
  }

  return 0;
}


int wrap_nvml_get_fanpcnt(wrap_nvml_handle *nvmlh,
                          int cudaindex, unsigned int *fanpcnt) {
  wrap_nvmlReturn_t rc;
  int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
  if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
    return -1;

  rc = nvmlh->nvmlDeviceGetFanSpeed(nvmlh->devs[gpuindex], fanpcnt);
  if (rc != WRAPNVML_SUCCESS) {
    return -1; 
  }

  return 0;
}


int wrap_nvml_get_power_usage(wrap_nvml_handle *nvmlh,
                              int cudaindex,
                              unsigned int *milliwatts) {
  int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
  if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
    return -1;

  if (nvmlh->nvmlDeviceGetPowerUsage(nvmlh->devs[gpuindex], milliwatts) != WRAPNVML_SUCCESS)
    return -1; 

  return 0;
}


#if defined(__cplusplus)
}
#endif


