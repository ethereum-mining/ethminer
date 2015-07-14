#include <CL/cl.h>
#include "param_struct.h"
#include <platform/icd_test_log.h>

extern cl_context context;

extern cl_platform_id  platform;

extern cl_device_id devices;

int ret_val;

struct clRetainContext_st clRetainContextData[NUM_ITEMS_clRetainContext] =
{
    {NULL}
};

struct clGetContextInfo_st clGetContextInfoData[NUM_ITEMS_clGetContextInfo] =
{
    {NULL, 0, 0, NULL, NULL}
};


struct clGetPlatformInfo_st clGetPlatformInfoData[NUM_ITEMS_clGetPlatformInfo] =
{
    {NULL, 0, 0, NULL, NULL}
};

struct clGetDeviceInfo_st clGetDeviceInfoData[NUM_ITEMS_clGetDeviceInfo] =
{
    {NULL, 0, 0, NULL, NULL}
};

struct clCreateSubDevices_st clCreateSubDevicesData[NUM_ITEMS_clCreateSubDevices] =
{
    {NULL, NULL, 0, NULL, NULL}
};


struct clRetainDevice_st clRetainDeviceData[NUM_ITEMS_clRetainDevice] =
{
    {NULL}
};


int test_clRetainContext(const struct clRetainContext_st* data)
{
    test_icd_app_log("clRetainContext(%p)\n", context); 

    ret_val = clRetainContext(context);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}



int test_clGetContextInfo(const struct clGetContextInfo_st* data)
{
    test_icd_app_log("clGetContextInfo(%p, %u, %u, %p, %p)\n",
                     context,
                     data->param_name, 
                     data->param_value_size,
                     data->param_value, 
                     data->param_value_size_ret); 


    ret_val = clGetContextInfo(context,
            data->param_name, 
            data->param_value_size,
            data->param_value, 
            data->param_value_size_ret);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

int test_clGetPlatformInfo(const struct clGetPlatformInfo_st* data)
{
    test_icd_app_log("clGetPlatformInfo(%p, %u, %u, %p, %p)\n",
                     platform,
                     data->param_name,
                     data->param_value_size,
                     data->param_value, 
                     data->param_value_size_ret); 					 

    ret_val = clGetPlatformInfo(platform,
            data->param_name,
            data->param_value_size,
            data->param_value,
            data->param_value_size_ret); 

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clGetDeviceInfo(const struct clGetDeviceInfo_st* data)
{ 
    test_icd_app_log("clGetDeviceInfo(%p, %u, %u, %p, %p)\n",
                     devices,
                     data->param_name, 
                     data->param_value_size, 
                     data->param_value, 
                     data->param_value_size_ret);		 

    ret_val = clGetDeviceInfo(devices,
            data->param_name, 
            data->param_value_size, 
            data->param_value, 
            data->param_value_size_ret);		 

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

int test_clCreateSubDevices(const struct clCreateSubDevices_st* data)
{
    test_icd_app_log("clCreateSubDevices(%p, %p, %u, %p, %p)\n",
                     devices,
                     data->properties, 
                     data->num_entries, 
                     &devices, 
                     data->num_devices); 

    ret_val = clCreateSubDevices(devices,
            data->properties, 
            data->num_entries,
            &devices, 
            data->num_devices); 

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

int test_clRetainDevice(const struct clRetainDevice_st* data)
{
    test_icd_app_log("clRetainDevice(%p)\n", devices); 

    ret_val = clRetainDevice(devices); 

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

int test_platforms()
{
    int i;

    for (i = 0;i<NUM_ITEMS_clRetainContext;i++) { 
        test_clRetainContext(&clRetainContextData[i]); 
    } 

    for (i = 0;i<NUM_ITEMS_clGetContextInfo;i++) { 
        test_clGetContextInfo(&clGetContextInfoData[i]); 
    } 

#if 0
    for (i = 0;i<NUM_ITEMS_clGetPlatformInfo;i++) { 
        test_clGetPlatformInfo(&clGetPlatformInfoData[i]); 
    }
#endif

    for (i = 0;i<NUM_ITEMS_clGetDeviceInfo;i++) { 
        test_clGetDeviceInfo(&clGetDeviceInfoData[i]); 
    } 

    for (i = 0;i<NUM_ITEMS_clCreateSubDevices;i++) { 
        test_clCreateSubDevices(&clCreateSubDevicesData[i]); 
    } 

    for (i = 0;i<NUM_ITEMS_clRetainDevice;i++) { 
        test_clRetainDevice(&clRetainDeviceData[i]); 
    } 

    return 0;
}
