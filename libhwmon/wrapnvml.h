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

#ifndef _WRAPNVML_H_
#define _WRAPNVML_H_

#if defined(__cplusplus)
extern "C" {
#endif

/* 
 * Ugly hacks to avoid dependencies on the real nvml.h until it starts
 * getting included with the CUDA toolkit or a GDK that's got a known 
 * install location, etc.
 */
typedef enum wrap_nvmlReturn_enum {
  WRAPNVML_SUCCESS = 0
} wrap_nvmlReturn_t;

typedef void * wrap_nvmlDevice_t;

/* our own version of the PCI info struct */
typedef struct {
  char bus_id_str[16];             /* string form of bus info */
  unsigned int domain;
  unsigned int bus;
  unsigned int device;
  unsigned int pci_device_id;      /* combined device and vendor id */
  unsigned int pci_subsystem_id;
  unsigned int res0;               /* NVML internal use only */
  unsigned int res1;
  unsigned int res2;
  unsigned int res3;
} wrap_nvmlPciInfo_t;


/* 
 * Handle to hold the function pointers for the entry points we need,
 * and the shared library itself.
 */
typedef struct {
  void *nvml_dll;
  int nvml_gpucount;
  int cuda_gpucount;
  unsigned int *nvml_pci_domain_id;
  unsigned int *nvml_pci_bus_id;
  unsigned int *nvml_pci_device_id;
  int *nvml_cuda_device_id;          /* map NVML dev to CUDA dev */
  int *cuda_nvml_device_id;          /* map CUDA dev to NVML dev */
  wrap_nvmlDevice_t *devs;
  wrap_nvmlReturn_t (*nvmlInit)(void);
  wrap_nvmlReturn_t (*nvmlDeviceGetCount)(int *);
  wrap_nvmlReturn_t (*nvmlDeviceGetHandleByIndex)(int, wrap_nvmlDevice_t *);
  wrap_nvmlReturn_t (*nvmlDeviceGetPciInfo)(wrap_nvmlDevice_t, wrap_nvmlPciInfo_t *);
  wrap_nvmlReturn_t (*nvmlDeviceGetName)(wrap_nvmlDevice_t, char *, int);
  wrap_nvmlReturn_t (*nvmlDeviceGetTemperature)(wrap_nvmlDevice_t, int, unsigned int *);
  wrap_nvmlReturn_t (*nvmlDeviceGetFanSpeed)(wrap_nvmlDevice_t, unsigned int *);
  wrap_nvmlReturn_t (*nvmlDeviceGetPowerUsage)(wrap_nvmlDevice_t, unsigned int *);
  wrap_nvmlReturn_t (*nvmlShutdown)(void);
} wrap_nvml_handle;


wrap_nvml_handle * wrap_nvml_create();
int wrap_nvml_destroy(wrap_nvml_handle *nvmlh);

/*
 * Query the number of GPUs seen by NVML
 */
int wrap_nvml_get_gpucount(wrap_nvml_handle *nvmlh, int *gpucount);

/*
 * Query the number of GPUs seen by CUDA 
 */
int wrap_cuda_get_gpucount(wrap_nvml_handle *nvmlh, int *gpucount);


/*
 * query the name of the GPU model from the CUDA device ID
 *
 */
int wrap_nvml_get_gpu_name(wrap_nvml_handle *nvmlh,
                           int gpuindex, 
                           char *namebuf,
                           int bufsize);

/* 
 * Query the current GPU temperature (Celsius), from the CUDA device ID
 */
int wrap_nvml_get_tempC(wrap_nvml_handle *nvmlh,
                        int gpuindex, unsigned int *tempC);

/* 
 * Query the current GPU fan speed (percent) from the CUDA device ID
 */
int wrap_nvml_get_fanpcnt(wrap_nvml_handle *nvmlh,
                          int gpuindex, unsigned int *fanpcnt);

/* 
 * Query the current GPU power usage in millwatts from the CUDA device ID
 *
 * This feature is only available on recent GPU generations and may be
 * limited in some cases only to Tesla series GPUs.
 * If the query is run on an unsupported GPU, this routine will return -1.
 */
int wrap_nvml_get_power_usage(wrap_nvml_handle *nvmlh,
                              int gpuindex,
                              unsigned int *milliwatts);


#if defined(__cplusplus)
}
#endif

#endif
