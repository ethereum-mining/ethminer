#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef CL_USE_DEPRECATED_OPENCL_1_0_APIS
#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#endif

#ifndef CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#endif

// Need to rename all CL API functions to prevent ICD loader functions calling
// themselves via the dispatch table. Include this before cl headers.
#include "rename_api.h"

#include <CL/cl.h>
#include <platform/icd_test_log.h>
#include "icd_structs.h"

#define CL_PLATFORM_ICD_SUFFIX_KHR                  0x0920
CL_API_ENTRY cl_int CL_API_CALL
clIcdGetPlatformIDsKHR(cl_uint, cl_platform_id *, cl_uint *);

struct _cl_platform_id
{
    CLIicdDispatchTable* dispatch;
    const char *profile;
    const char *version;
    const char *name;
    const char *vendor;
    const char *extensions;
    const char *suffix;
};

struct _cl_device_id
{
    CLIicdDispatchTable* dispatch;
};

struct _cl_context
{
    CLIicdDispatchTable* dispatch;
};

struct _cl_command_queue
{
    CLIicdDispatchTable* dispatch;
};

struct _cl_mem
{
    CLIicdDispatchTable* dispatch;
};

struct _cl_program
{
    CLIicdDispatchTable* dispatch;
};

struct _cl_kernel
{
    CLIicdDispatchTable* dispatch;
};

struct _cl_event
{
    CLIicdDispatchTable* dispatch;
};

struct _cl_sampler
{
    CLIicdDispatchTable* dispatch;
};

static CLIicdDispatchTable* dispatchTable = NULL;
static cl_platform_id platform = NULL;
static cl_bool initialized = CL_FALSE;

CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformIDs(cl_uint           num_entries ,
                 cl_platform_id *  platforms ,
                 cl_uint *         num_platforms) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clGetPlatformIDs(%u, %p, %p)\n",
                      num_entries,
                      platforms,
                      num_platforms);
    return_value = clIcdGetPlatformIDsKHR(num_entries, platforms, num_platforms);
    test_icd_stub_log("Value returned: %d\n", return_value); 
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformInfo(cl_platform_id    platform,
                  cl_platform_info  param_name,
                  size_t            param_value_size,
                  void *            param_value,
                  size_t *          param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int result = CL_SUCCESS;
    cl_int return_value = CL_SUCCESS;
    const char *returnString = NULL;
    size_t returnStringLength = 0;
    /*test_icd_stub_log("clGetPlatformInfo(%p, %u, %u, %p, %p)\n", 
                      platform, 
                      param_name, 
                      param_value_size, 
                      param_value, 
                      param_value_size_ret);*/

    // validate the arguments
    if (param_value_size == 0 && param_value != NULL) {
        return CL_INVALID_VALUE;
    }
    // select the string to return
    switch(param_name) {
        case CL_PLATFORM_PROFILE:
            returnString = platform->profile;
            break;
        case CL_PLATFORM_VERSION:
            returnString = platform->version;
            break;
        case CL_PLATFORM_NAME:
            returnString = platform->name;
            break;
        case CL_PLATFORM_VENDOR:
            returnString = platform->vendor;
            break;
        case CL_PLATFORM_EXTENSIONS:
            returnString = platform->extensions;
            break;
        case CL_PLATFORM_ICD_SUFFIX_KHR:
            returnString = platform->suffix;
            break;
        default:
            /*test_icd_stub_log("Value returned: %d\n", 
                                CL_INVALID_VALUE);*/
            return CL_INVALID_VALUE;
            break;
    }

    // make sure the buffer passed in is big enough for the result
    returnStringLength = strlen(returnString)+1;
    if (param_value_size && param_value_size < returnStringLength) {
        /*test_icd_stub_log("Value returned: %d\n", 
                          CL_INVALID_VALUE);*/
        return CL_INVALID_VALUE;
    }

    // pass the data back to the user
    if (param_value) {
        memcpy(param_value, returnString, returnStringLength);
    }
    if (param_value_size_ret) {
        *param_value_size_ret = returnStringLength;
    }

    /*test_icd_stub_log("Value returned: %d\n",
                      return_value);*/
    return return_value;
}


/* Device APIs */
CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDs(cl_platform_id   platform,
               cl_device_type   device_type,
               cl_uint          num_entries,
               cl_device_id *   devices,
               cl_uint *        num_devices) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_SUCCESS;

    if ((num_entries > 1 || num_entries < 0) && devices != NULL) {
        return_value = CL_INVALID_VALUE;
    }
    else {
        cl_device_id obj = (cl_device_id) malloc(sizeof(cl_device_id));
        obj->dispatch = dispatchTable;
        *devices = obj;
    }
    if (num_devices) {
        *num_devices = 1;
    }

    test_icd_stub_log("clGetDeviceIDs(%p, %x, %u, %p, %p)\n",
                      platform,
                      device_type,
                      num_entries,
                      devices,
                      num_devices);
    test_icd_stub_log("Value returned: %d\n", CL_SUCCESS);
    return CL_SUCCESS;
}



CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceInfo(cl_device_id    device,
                cl_device_info  param_name,
                size_t          param_value_size,
                void *          param_value,
                size_t *        param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clGetDeviceInfo(%p, %u, %u, %p, %p)\n",
                      device,
                      param_name,
                      param_value_size,
                      param_value,
                      param_value_size_ret);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clCreateSubDevices(cl_device_id in_device,
                   const cl_device_partition_property *properties,
                   cl_uint num_entries,
                   cl_device_id *out_devices,
                   cl_uint *num_devices) CL_API_SUFFIX__VERSION_1_2
{

    cl_int return_value = CL_OUT_OF_RESOURCES;

    test_icd_stub_log("clCreateSubDevices(%p, %p, %u, %p, %p)\n",
                      in_device,
                      properties,
                      num_entries,
                      out_devices,
                      num_devices);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}


CL_API_ENTRY cl_int CL_API_CALL
clRetainDevice(cl_device_id device) CL_API_SUFFIX__VERSION_1_2
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clRetainDevice(%p)\n", device);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}


CL_API_ENTRY cl_int CL_API_CALL
clReleaseDevice(cl_device_id device) CL_API_SUFFIX__VERSION_1_2

{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clReleaseDevice(%p)\n", device);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}


/* Context APIs  */
CL_API_ENTRY cl_context CL_API_CALL
clCreateContext(const cl_context_properties * properties,
                cl_uint                       num_devices ,
                const cl_device_id *          devices,
                void (CL_CALLBACK * pfn_notify)(const char *, const void *, size_t, void *),
                void *                        user_data,
                cl_int *                      errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_context obj = (cl_context) malloc(sizeof(struct _cl_context));
    obj->dispatch = dispatchTable;
    test_icd_stub_log("clCreateContext(%p, %u, %p, %p, %p, %p)\n",
                      properties,
                      num_devices,
                      devices,
                      pfn_notify,
                      user_data,
                      errcode_ret);
    pfn_notify(NULL, NULL, 0, NULL);
    test_icd_stub_log("createcontext_callback(%p, %p, %u, %p)\n",
                      NULL,
                      NULL,
                      0,
                      NULL);

    test_icd_stub_log("Value returned: %p\n", obj);
    return obj;
}


CL_API_ENTRY cl_context CL_API_CALL
clCreateContextFromType(const cl_context_properties * properties,
                        cl_device_type                device_type,
                        void (CL_CALLBACK *     pfn_notify)(const char *, const void *, size_t, void *),
                        void *                        user_data,
                        cl_int *                      errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_context obj = (cl_context) malloc(sizeof(struct _cl_context));
    obj->dispatch = dispatchTable;
    test_icd_stub_log("clCreateContextFromType(%p, %x, %p, %p, %p)\n",
                      properties,
                      device_type,
                      pfn_notify,
                      user_data,
                      errcode_ret);
    pfn_notify(NULL, NULL, 0, NULL);

    test_icd_stub_log ("createcontext_callback(%p, %p, %u, %p)\n", 
                       NULL, 
                       NULL, 
                       0, 
                       NULL);
    
    test_icd_stub_log("Value returned: %p\n", 
		              obj);
    return obj;
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainContext(cl_context context) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clRetainContext(%p)\n", context);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseContext(cl_context context) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clReleaseContext(%p)\n", context);
    free(context);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetContextInfo(cl_context         context,
                 cl_context_info    param_name,
                 size_t             param_value_size,
                 void *             param_value,
                 size_t *           param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clGetContextInfo(%p, %u, %u, %p, %p)\n",
                      context,
                      param_name,
                      param_value_size,
                      param_value,
                      param_value_size_ret);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

/* Command Queue APIs */
CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueue(cl_context                     context,
                     cl_device_id                   device,
                     cl_command_queue_properties    properties,
                     cl_int *                       errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_command_queue obj = (cl_command_queue) malloc(sizeof(struct _cl_command_queue));
    obj->dispatch = dispatchTable;
    test_icd_stub_log("clCreateCommandQueue(%p, %p, %x, %p)\n",
                      context,
                      device,
                      properties,
                      errcode_ret);

    test_icd_stub_log("Value returned: %p\n", obj);
    return obj;
}

CL_API_ENTRY cl_int CL_API_CALL
clSetCommandQueueProperty(cl_command_queue               command_queue ,
                            cl_command_queue_properties    properties ,
                            cl_bool                        enable ,
                            cl_command_queue_properties *  old_properties) CL_EXT_SUFFIX__VERSION_1_0_DEPRECATED
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clSetCommandQueueProperty(%p, %p, %u, %p)\n",
                      command_queue,
                      properties,
                      enable,
                      old_properties);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainCommandQueue(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clRetainCommandQueue(%p)\n", command_queue);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseCommandQueue(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clReleaseCommandQueue(%p)\n", command_queue);
    free(command_queue);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetCommandQueueInfo(cl_command_queue       command_queue ,
                      cl_command_queue_info  param_name ,
                      size_t                 param_value_size ,
                      void *                 param_value ,
                      size_t *               param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clGetCommandQueueInfo(%p, %u, %u, %p, %p)\n",
                      command_queue,
                      param_name,
                      param_value_size,
                      param_value,
                      param_value_size_ret);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}



/* Memory Object APIs */
CL_API_ENTRY cl_mem CL_API_CALL
clCreateBuffer(cl_context    context ,
               cl_mem_flags  flags ,
               size_t        size ,
               void *        host_ptr ,
               cl_int *      errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_mem obj = (cl_mem) malloc(sizeof(struct _cl_mem));
    obj->dispatch = dispatchTable;
    test_icd_stub_log("clCreateBuffer(%p, %x, %u, %p, %p)\n",
                      context,
                      flags,
                      size,
                      host_ptr,
                      errcode_ret);

    test_icd_stub_log("Value returned: %p\n", obj);
    return obj;
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateSubBuffer(cl_mem                    buffer ,
                  cl_mem_flags              flags ,
                  cl_buffer_create_type     buffer_create_type ,
                  const void *              buffer_create_info ,
                  cl_int *                  errcode_ret) CL_API_SUFFIX__VERSION_1_1
{
    cl_mem obj = (cl_mem) malloc(sizeof(struct _cl_mem));
    obj->dispatch = dispatchTable;
    test_icd_stub_log("clCreateSubBuffer(%p, %x, %u, %p, %p)\n",
                      buffer,
                      flags,
                      buffer_create_type,
                      buffer_create_info,
                      errcode_ret);

    test_icd_stub_log("Value returned: %p\n", obj);
    return obj;
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateImage(cl_context              context,
                            cl_mem_flags            flags,
                            const cl_image_format * image_format,
                            const cl_image_desc *   image_desc,
                            void *                  host_ptr,
                            cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_2
{
    cl_mem obj = (cl_mem) malloc(sizeof(struct _cl_mem));
    obj->dispatch = dispatchTable;
    test_icd_stub_log("clCreateImage(%p, %x, %p, %p, %p, %p)\n",
                      context,
                      flags,
                      image_format,
                      image_desc,
                      host_ptr,
                      errcode_ret);

    test_icd_stub_log("Value returned: %p\n", obj);
    return obj;
}


CL_API_ENTRY cl_mem CL_API_CALL
clCreateImage2D(cl_context              context ,
                cl_mem_flags            flags ,
                const cl_image_format * image_format ,
                size_t                  image_width ,
                size_t                  image_height ,
                size_t                  image_row_pitch ,
                void *                  host_ptr ,
                cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_mem obj = (cl_mem) malloc(sizeof(struct _cl_mem));
    obj->dispatch = dispatchTable;
    test_icd_stub_log("clCreateImage2D(%p, %x, %p, %u, %u, %u, %p, %p)\n",
                      context,
                      flags,
                      image_format,
                      image_width,
                      image_height,
                      image_row_pitch,
                      host_ptr);

    test_icd_stub_log("Value returned: %p\n", obj);
    return obj;
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateImage3D(cl_context              context,
                cl_mem_flags            flags,
                const cl_image_format * image_format,
                size_t                  image_width,
                size_t                  image_height ,
                size_t                  image_depth ,
                size_t                  image_row_pitch ,
                size_t                  image_slice_pitch ,
                void *                  host_ptr ,
                cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_mem obj = (cl_mem) malloc(sizeof(struct _cl_mem));
    obj->dispatch = dispatchTable;
    test_icd_stub_log("clCreateImage3D(%p, %x, %p, %u, %u, %u, %u, %u, %p, %p)\n",
                      context,
                      flags,
                      image_format,
                      image_width,
                      image_height,
                      image_depth,
                      image_row_pitch,
                      image_slice_pitch,
                      host_ptr,
                      errcode_ret);

    test_icd_stub_log("Value returned: %p\n", obj);
    return obj;
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainMemObject(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clRetainMemObject(%p)\n", memobj);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseMemObject(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clReleaseMemObject(%p)\n", memobj);
    free(memobj);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetSupportedImageFormats(cl_context           context,
                           cl_mem_flags         flags,
                           cl_mem_object_type   image_type ,
                           cl_uint              num_entries ,
                           cl_image_format *    image_formats ,
                           cl_uint *            num_image_formats) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clGetSupportedImageFormats(%p, %x, %u, %u, %p, %p)\n",
                      context,
                      flags,
                      image_type,
                      num_entries,
                      image_formats,
                      num_image_formats);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetMemObjectInfo(cl_mem            memobj ,
                   cl_mem_info       param_name ,
                   size_t            param_value_size ,
                   void *            param_value ,
                   size_t *          param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clGetMemObjectInfo(%p, %u, %u, %p, %p)\n",
                      memobj,
                      param_name,
                      param_value_size,
                      param_value,
                      param_value_size_ret);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetImageInfo(cl_mem            image ,
               cl_image_info     param_name ,
               size_t            param_value_size ,
               void *            param_value ,
               size_t *          param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clGetImageInfo(%p, %u, %u, %p, %p)\n",
                      image,
                      param_name,
                      param_value_size,
                      param_value,
                      param_value_size_ret);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clSetMemObjectDestructorCallback(cl_mem  memobj ,
                                    void (CL_CALLBACK * pfn_notify)(cl_mem  memobj , void* user_data),
                                    void * user_data)             CL_API_SUFFIX__VERSION_1_1
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clSetMemObjectDestructorCallback(%p, %p, %p)\n",
                      memobj,
                      pfn_notify,
                      user_data);
    pfn_notify(memobj, NULL);
    test_icd_stub_log("setmemobjectdestructor_callback(%p, %p)\n",
               memobj,
               NULL);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

/* Sampler APIs  */
CL_API_ENTRY cl_sampler CL_API_CALL
clCreateSampler(cl_context           context ,
                cl_bool              normalized_coords ,
                cl_addressing_mode   addressing_mode ,
                cl_filter_mode       filter_mode ,
                cl_int *             errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_sampler obj = (cl_sampler) malloc(sizeof(struct _cl_sampler));
    obj->dispatch = dispatchTable;
    test_icd_stub_log("clCreateSampler(%p, %u, %u, %u, %p)\n",
                      context,
                      normalized_coords,
                      addressing_mode,
                      filter_mode,
                      errcode_ret);

    test_icd_stub_log("Value returned: %p\n", obj);
    return obj;
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainSampler(cl_sampler  sampler) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clRetainSampler(%p)\n", sampler);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseSampler(cl_sampler  sampler) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clReleaseSampler(%p)\n", sampler);
    free(sampler);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetSamplerInfo(cl_sampler          sampler ,
                 cl_sampler_info     param_name ,
                 size_t              param_value_size ,
                 void *              param_value ,
                 size_t *            param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clGetSamplerInfo(%p, %u, %u, %p, %p)\n",
                      sampler,
                      param_name,
                      param_value_size,
                      param_value,
                      param_value_size_ret);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

/* Program Object APIs  */
CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithSource(cl_context         context ,
                          cl_uint            count ,
                          const char **      strings ,
                          const size_t *     lengths ,
                          cl_int *           errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_program obj = (cl_program) malloc(sizeof(struct _cl_program));
    obj->dispatch = dispatchTable;
    test_icd_stub_log("clCreateProgramWithSource(%p, %u, %p, %p, %p)\n",
                      context,
                      count,
                      strings,
                      lengths,
                      errcode_ret);

    test_icd_stub_log("Value returned: %p\n", obj);
    return obj;
}

CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithBinary(cl_context                      context ,
                          cl_uint                         num_devices ,
                          const cl_device_id *            device_list ,
                          const size_t *                  lengths ,
                          const unsigned char **          binaries ,
                          cl_int *                        binary_status ,
                          cl_int *                        errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_program obj = (cl_program) malloc(sizeof(struct _cl_program));
    obj->dispatch = dispatchTable;
    test_icd_stub_log("clCreateProgramWithBinary(%p, %u, %p, %p, %p, %p, %p)\n",
                      context,
                      num_devices,
                      device_list,
                      lengths,
                      binaries,
                      binary_status,
                      errcode_ret);

    test_icd_stub_log("Value returned: %p\n", obj);
    return obj;
}

CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithBuiltInKernels(cl_context             context ,
                                  cl_uint                num_devices ,
                                  const cl_device_id *   device_list ,
                                  const char *           kernel_names ,
                                  cl_int *               errcode_ret) CL_API_SUFFIX__VERSION_1_2
{
    cl_program obj = (cl_program) malloc(sizeof(struct _cl_program));
    obj->dispatch = dispatchTable;
    test_icd_stub_log("clCreateProgramWithBuiltInKernels(%p, %u, %p, %p, %p)\n",
                      context,
                      num_devices,
                      device_list,
                      kernel_names,
                      errcode_ret);

    test_icd_stub_log("Value returned: %p\n", obj);
    return obj;
}


CL_API_ENTRY cl_int CL_API_CALL
clRetainProgram(cl_program  program) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clRetainProgram(%p)\n",
                      program);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseProgram(cl_program  program) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clReleaseProgram(%p)\n", program);
    free(program);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clBuildProgram(cl_program            program ,
               cl_uint               num_devices ,
               const cl_device_id *  device_list ,
               const char *          options ,
               void (CL_CALLBACK *   pfn_notify)(cl_program  program , void *  user_data),
               void *                user_data) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clBuildProgram(%p, %u, %p, %p, %p, %p)\n",
                      program,
                      num_devices,
                      device_list,
                      options,
                      pfn_notify,
                      user_data);
    pfn_notify(program, NULL);
    test_icd_stub_log("program_callback(%p, %p)\n", program, NULL);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clUnloadCompiler(void) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clUnloadCompiler()\n");
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clCompileProgram(cl_program            program ,
                 cl_uint               num_devices ,
                 const cl_device_id *  device_list ,
                 const char *          options ,
                 cl_uint               num_input_headers ,
                 const cl_program *    input_headers,
                 const char **         header_include_names ,
                 void (CL_CALLBACK *   pfn_notify)(cl_program  program , void *  user_data),
                 void *                user_data) CL_API_SUFFIX__VERSION_1_2
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clCompileProgram(%p, %u, %p, %p, %u, %p, %p, %p)\n",
                      program,
                      num_devices,
                      device_list,
                      options,
                      num_input_headers,
                      header_include_names,
                      pfn_notify,
                      user_data);
    pfn_notify(program, NULL);
    test_icd_stub_log("program_callback(%p, %p)\n", program, NULL);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_program CL_API_CALL
clLinkProgram(cl_context            context ,
              cl_uint               num_devices ,
              const cl_device_id *  device_list ,
              const char *          options ,
              cl_uint               num_input_programs ,
              const cl_program *    input_programs ,
              void (CL_CALLBACK *   pfn_notify)(cl_program  program , void *  user_data),
              void *                user_data ,
              cl_int *              errcode_ret) CL_API_SUFFIX__VERSION_1_2
{
    cl_program obj = (cl_program) malloc(sizeof(cl_program));
    obj->dispatch = dispatchTable;
    test_icd_stub_log("clLinkProgram(%p, %u, %p, %p, %u, %p, %p, %p, %p)\n",
                      context,
                      num_devices,
                      device_list,
                      options,
                      num_input_programs,
                      input_programs,
                      pfn_notify,
                      user_data,
                      errcode_ret);
    pfn_notify(obj, NULL);
    test_icd_stub_log("program_callback(%p, %p)\n", obj, NULL);
    test_icd_stub_log("Value returned: %p\n", obj);
    return obj;
}


CL_API_ENTRY cl_int CL_API_CALL
clUnloadPlatformCompiler(cl_platform_id  platform) CL_API_SUFFIX__VERSION_1_2
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clUnloadPlatformCompiler(%p)\n", platform);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetProgramInfo(cl_program          program ,
                 cl_program_info     param_name ,
                 size_t              param_value_size ,
                 void *              param_value ,
                 size_t *            param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clGetProgramInfo(%p, %u, %u, %p, %p)\n",
                      program,
                      param_name,
                      param_value_size,
                      param_value,
                      param_value_size_ret);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetProgramBuildInfo(cl_program             program ,
                      cl_device_id           device ,
                      cl_program_build_info  param_name ,
                      size_t                 param_value_size ,
                      void *                 param_value ,
                      size_t *               param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clGetProgramBuildInfo(%p, %p, %u, %u, %p, %p)\n",
                      program,
                      device,
                      param_name,
                      param_value_size,
                      param_value,
                      param_value_size_ret);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

/* Kernel Object APIs */
CL_API_ENTRY cl_kernel CL_API_CALL
clCreateKernel(cl_program       program ,
               const char *     kernel_name ,
               cl_int *         errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_kernel obj = (cl_kernel) malloc(sizeof(struct _cl_kernel));
    obj->dispatch = dispatchTable;
    test_icd_stub_log("clCreateKernel(%p, %p, %p)\n",
                      program,
                      kernel_name,
                      errcode_ret);

    test_icd_stub_log("Value returned: %p\n", obj);
    return obj;
}

CL_API_ENTRY cl_int CL_API_CALL
clCreateKernelsInProgram(cl_program      program ,
                         cl_uint         num_kernels ,
                         cl_kernel *     kernels ,
                         cl_uint *       num_kernels_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clCreateKernelsInProgram(%p, %u, %p, %p)\n",
                      program,
                      num_kernels,
                      kernels,
                      num_kernels_ret);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainKernel(cl_kernel     kernel) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clRetainKernel(%p)\n", kernel);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseKernel(cl_kernel    kernel) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clReleaseKernel(%p)\n", kernel);
    free(kernel);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArg(cl_kernel     kernel ,
               cl_uint       arg_index ,
               size_t        arg_size ,
               const void *  arg_value) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clSetKernelArg(%p, %u, %u, %p)\n",
                      kernel,
                      arg_index,
                      arg_size,
                      arg_value);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetKernelInfo(cl_kernel        kernel ,
                cl_kernel_info   param_name ,
                size_t           param_value_size ,
                void *           param_value ,
                size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clGetKernelInfo(%p, %u, %u, %p, %p)\n",
                      kernel,
                      param_name,
                      param_value_size,
                      param_value,
                      param_value_size_ret);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetKernelArgInfo(cl_kernel        kernel ,
                   cl_uint          arg_indx ,
                   cl_kernel_arg_info   param_name ,
                   size_t           param_value_size ,
                   void *           param_value ,
                   size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_2
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clGetKernelArgInfo(%p, %u, %u, %u, %p, %p)\n",
                      kernel,
                      arg_indx,
                      param_name,
                      param_value_size,
                      param_value,
                      param_value_size_ret);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetKernelWorkGroupInfo(cl_kernel                   kernel ,
                         cl_device_id                device ,
                         cl_kernel_work_group_info   param_name ,
                         size_t                      param_value_size ,
                         void *                      param_value ,
                         size_t *                    param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clGetKernelWorkGroupInfo(%p, %p, %u, %u, %p, %p)\n",
                      kernel,
                      device,
                      param_name,
                      param_value_size,
                      param_value,
                      param_value_size_ret);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

/* Event Object APIs  */
CL_API_ENTRY cl_int CL_API_CALL
clWaitForEvents(cl_uint              num_events ,
                const cl_event *     event_list) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clWaitForEvents(%u, %p)\n",
                      num_events,
                      event_list);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetEventInfo(cl_event          event ,
               cl_event_info     param_name ,
               size_t            param_value_size ,
               void *            param_value ,
               size_t *          param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clGetEventInfo(%p, %u, %u, %p, %p)\n",
                      event,
                      param_name,
                      param_value_size,
                      param_value,
                      param_value_size_ret);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_event CL_API_CALL
clCreateUserEvent(cl_context     context ,
                  cl_int *       errcode_ret) CL_API_SUFFIX__VERSION_1_1
{
    cl_event obj = (cl_event) malloc(sizeof(struct _cl_event));
    obj->dispatch = dispatchTable;
    test_icd_stub_log("clCreateUserEvent(%p, %p)\n", context, errcode_ret);
    test_icd_stub_log("Value returned: %p\n", obj);
    return obj;
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainEvent(cl_event  event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clRetainEvent(%p)\n", event);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseEvent(cl_event  event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clReleaseEvent(%p)\n", event);
    free(event);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clSetUserEventStatus(cl_event    event ,
                     cl_int      execution_status) CL_API_SUFFIX__VERSION_1_1
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clSetUserEventStatus(%p, %d)\n",
                      event,
                      execution_status);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clSetEventCallback(cl_event     event ,
                    cl_int       command_exec_callback_type ,
                    void (CL_CALLBACK *  pfn_notify)(cl_event, cl_int, void *),
                    void *       user_data) CL_API_SUFFIX__VERSION_1_1
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clSetEventCallback(%p, %d, %p, %p)\n",
                      event,
                      command_exec_callback_type,
                      pfn_notify,
                      user_data);
    pfn_notify(event, command_exec_callback_type, NULL);
    test_icd_stub_log("setevent_callback(%p, %d, %p)\n",
                      event,
                      command_exec_callback_type,
                      NULL);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

/* Profiling APIs  */
CL_API_ENTRY cl_int CL_API_CALL
clGetEventProfilingInfo(cl_event             event ,
                        cl_profiling_info    param_name ,
                        size_t               param_value_size ,
                        void *               param_value ,
                        size_t *             param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clGetEventProfilingInfo(%p, %u, %u, %p, %p)\n",
                      event,
                      param_name,
                      param_value_size,
                      param_value,
                      param_value_size_ret);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

/* Flush and Finish APIs */
CL_API_ENTRY cl_int CL_API_CALL
clFlush(cl_command_queue  command_queue) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clFlush(%p)\n", command_queue);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clFinish(cl_command_queue  command_queue) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clFinish(%p)\n", command_queue);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

/* Enqueued Commands APIs */
CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBuffer(cl_command_queue     command_queue ,
                    cl_mem               buffer ,
                    cl_bool              blocking_read ,
                    size_t               offset ,
                    size_t               cb ,
                    void *               ptr ,
                    cl_uint              num_events_in_wait_list ,
                    const cl_event *     event_wait_list ,
                    cl_event *           event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueReadBuffer(%p, %p, %u, %u, %u, %p, %u, %p, %p)\n",
                      command_queue,
                      buffer,
                      blocking_read,
                      offset,
                      cb,
                      ptr,
                      num_events_in_wait_list,
                      event_wait_list, event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBufferRect(cl_command_queue     command_queue ,
                        cl_mem               buffer ,
                        cl_bool              blocking_read ,
                        const size_t *       buffer_origin ,
                        const size_t *       host_origin ,
                        const size_t *       region ,
                        size_t               buffer_row_pitch ,
                        size_t               buffer_slice_pitch ,
                        size_t               host_row_pitch ,
                        size_t               host_slice_pitch ,
                        void *               ptr ,
                        cl_uint              num_events_in_wait_list ,
                        const cl_event *     event_wait_list ,
                        cl_event *           event) CL_API_SUFFIX__VERSION_1_1
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueReadBufferRect(%p, %p, %u, %p, %p, %p, %u, %u, %u, %u, %p, %u, %p, %p)\n",
                      command_queue,
                      buffer,
                      blocking_read,
                      buffer_origin,
                      host_origin,
                      region,
                      buffer_row_pitch,
                      buffer_slice_pitch,
                      host_row_pitch,
                      host_slice_pitch,
                      ptr,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBuffer(cl_command_queue    command_queue ,
                     cl_mem              buffer ,
                     cl_bool             blocking_write ,
                     size_t              offset ,
                     size_t              cb ,
                     const void *        ptr ,
                     cl_uint             num_events_in_wait_list ,
                     const cl_event *    event_wait_list ,
                     cl_event *          event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueWriteBuffer(%p, %p, %u, %u, %u, %p, %u, %p, %p)\n",
                      command_queue,
                      buffer,
                      blocking_write,
                      offset,
                      cb,
                      ptr,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBufferRect(cl_command_queue     command_queue ,
                         cl_mem               buffer ,
                         cl_bool              blocking_write ,
                         const size_t *       buffer_origin ,
                         const size_t *       host_origin ,
                         const size_t *       region ,
                         size_t               buffer_row_pitch ,
                         size_t               buffer_slice_pitch ,
                         size_t               host_row_pitch ,
                         size_t               host_slice_pitch ,
                         const void *         ptr ,
                         cl_uint              num_events_in_wait_list ,
                         const cl_event *     event_wait_list ,
                         cl_event *           event) CL_API_SUFFIX__VERSION_1_1
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueWriteBufferRect(%p, %p, %u, %p, %p, %p, %u, %u, %u, %u, %p, %u, %p, %p)\n",
                      command_queue,
                      buffer,
                      blocking_write,
                      buffer_origin,
                      host_origin,
                      region,
                      buffer_row_pitch,
                      buffer_slice_pitch,
                      host_row_pitch,
                      host_slice_pitch,
                      ptr,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBuffer(cl_command_queue     command_queue ,
                    cl_mem               src_buffer ,
                    cl_mem               dst_buffer ,
                    size_t               src_offset ,
                    size_t               dst_offset ,
                    size_t               cb ,
                    cl_uint              num_events_in_wait_list ,
                    const cl_event *     event_wait_list ,
                    cl_event *           event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueCopyBuffer(%p, %p, %p, %u, %u, %u, %u, %p, %p)\n",
                      command_queue,
                      src_buffer,
                      dst_buffer,
                      src_offset,
                      dst_offset,
                      cb,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBufferRect(cl_command_queue     command_queue ,
                        cl_mem               src_buffer ,
                        cl_mem               dst_buffer ,
                        const size_t *       src_origin ,
                        const size_t *       dst_origin ,
                        const size_t *       region ,
                        size_t               src_row_pitch ,
                        size_t               src_slice_pitch ,
                        size_t               dst_row_pitch ,
                        size_t               dst_slice_pitch ,
                        cl_uint              num_events_in_wait_list ,
                        const cl_event *     event_wait_list ,
                        cl_event *           event) CL_API_SUFFIX__VERSION_1_1
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueCopyBufferRect(%p, %p, %p, %p, %p, %p, %u, %u, %u, %u, %u, %p, %p)\n",
                      command_queue,
                      src_buffer,
                      dst_buffer,
                      src_origin,
                      dst_origin,
                      region,
                      src_row_pitch,
                      src_slice_pitch,
                      dst_row_pitch,
                      dst_slice_pitch,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}


CL_API_ENTRY cl_int CL_API_CALL
clEnqueueFillBuffer(cl_command_queue    command_queue ,
                    cl_mem              buffer ,
                    const void *        pattern ,
                    size_t              pattern_size ,
                    size_t              offset ,
                    size_t              cb ,
                    cl_uint             num_events_in_wait_list ,
                    const cl_event *    event_wait_list ,
                    cl_event *          event) CL_API_SUFFIX__VERSION_1_2
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueFillBuffer(%p, %p, %p, %u, %u, %u, %u, %p, %p)\n",
                      command_queue,
                      buffer,
                      pattern,
                      pattern_size,
                      offset,
                      cb,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}


CL_API_ENTRY cl_int CL_API_CALL
clEnqueueFillImage(cl_command_queue    command_queue ,
                   cl_mem              image ,
                   const void *        fill_color ,
                   const size_t *      origin ,
                   const size_t *      region ,
                   cl_uint             num_events_in_wait_list ,
                   const cl_event *    event_wait_list ,
                   cl_event *          event) CL_API_SUFFIX__VERSION_1_2
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueFillImage(%p, %p, %p, %p, %p, %u, %p, %p)\n",
                      command_queue,
                      image,
                      fill_color,
                      origin,
                      region,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}


CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadImage(cl_command_queue      command_queue ,
                   cl_mem                image ,
                   cl_bool               blocking_read ,
                   const size_t *        origin ,
                   const size_t *        region ,
                   size_t                row_pitch ,
                   size_t                slice_pitch ,
                   void *                ptr ,
                   cl_uint               num_events_in_wait_list ,
                   const cl_event *      event_wait_list ,
                   cl_event *            event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueReadImage(%p, %p, %u, %p, %p, %u, %u, %p, %u, %p, %p)\n",
                      command_queue,
                      image,
                      blocking_read,
                      origin,
                      region,
                      row_pitch,
                      slice_pitch,
                      ptr,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteImage(cl_command_queue     command_queue ,
                    cl_mem               image ,
                    cl_bool              blocking_write ,
                    const size_t *       origin ,
                    const size_t *       region ,
                    size_t               input_row_pitch ,
                    size_t               input_slice_pitch ,
                    const void *         ptr ,
                    cl_uint              num_events_in_wait_list ,
                    const cl_event *     event_wait_list ,
                    cl_event *           event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueWriteImage(%p, %p, %u, %p, %p, %u, %u, %p, %u, %p, %p)\n",
                      command_queue,
                      image,
                      blocking_write,
                      origin,
                      region,
                      input_row_pitch,
                      input_slice_pitch,
                      ptr,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyImage(cl_command_queue      command_queue ,
                   cl_mem                src_image ,
                   cl_mem                dst_image ,
                   const size_t *        src_origin ,
                   const size_t *        dst_origin ,
                   const size_t *        region ,
                   cl_uint               num_events_in_wait_list ,
                   const cl_event *      event_wait_list ,
                   cl_event *            event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueCopyImage(%p, %p, %p, %p, %p, %p, %u, %p, %p)\n",
                      command_queue,
                      src_image,
                      dst_image,
                      src_origin,
                      dst_origin,
                      region,
                      num_events_in_wait_list,
                      event_wait_list ,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyImageToBuffer(cl_command_queue  command_queue ,
                           cl_mem            src_image ,
                           cl_mem            dst_buffer ,
                           const size_t *    src_origin ,
                           const size_t *    region ,
                           size_t            dst_offset ,
                           cl_uint           num_events_in_wait_list ,
                           const cl_event *  event_wait_list ,
                           cl_event *        event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueCopyImageToBuffer(%p, %p, %p, %p, %p, %u, %u, %p, %p)\n",
                      command_queue,
                      src_image,
                      dst_buffer,
                      src_origin,
                      region,
                      dst_offset,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBufferToImage(cl_command_queue  command_queue ,
                           cl_mem            src_buffer ,
                           cl_mem            dst_image ,
                           size_t            src_offset ,
                           const size_t *    dst_origin ,
                           const size_t *    region ,
                           cl_uint           num_events_in_wait_list ,
                           const cl_event *  event_wait_list ,
                           cl_event *        event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueCopyBufferToImage(%p, %p, %p, %u, %p, %p, %u, %p, %p)\n",
                      command_queue,
                      src_buffer,
                      dst_image,
                      src_offset,
                      dst_origin,
                      region,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY void * CL_API_CALL
clEnqueueMapBuffer(cl_command_queue  command_queue ,
                   cl_mem            buffer ,
                   cl_bool           blocking_map ,
                   cl_map_flags      map_flags ,
                   size_t            offset ,
                   size_t            cb ,
                   cl_uint           num_events_in_wait_list ,
                   const cl_event *  event_wait_list ,
                   cl_event *        event ,
                   cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    void *return_value = (void *) malloc(sizeof(void *));
    test_icd_stub_log("clEnqueueMapBuffer(%p, %p, %u, %x, %u, %u, %u, %p, %p, %p)\n",
                      command_queue,
                      buffer,
                      blocking_map,
                      map_flags,
                      offset,
                      cb,
                      num_events_in_wait_list,
                      event_wait_list,
                      event,
                      errcode_ret);

    test_icd_stub_log("Value returned: %p\n", return_value);
    return return_value;
}

CL_API_ENTRY void * CL_API_CALL
clEnqueueMapImage(cl_command_queue   command_queue ,
                  cl_mem             image ,
                  cl_bool            blocking_map ,
                  cl_map_flags       map_flags ,
                  const size_t *     origin ,
                  const size_t *     region ,
                  size_t *           image_row_pitch ,
                  size_t *           image_slice_pitch ,
                  cl_uint            num_events_in_wait_list ,
                  const cl_event *   event_wait_list ,
                  cl_event *         event ,
                  cl_int *           errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    void *return_value = (void *) malloc(sizeof(void *));
    test_icd_stub_log("clEnqueueMapImage(%p, %p, %u, %x, %p, %p, %p, %p, %u, %p, %p, %p)\n",
                      command_queue,
                      image,
                      blocking_map,
                      map_flags,
                      origin,
                      region,
                      image_row_pitch,
                      image_slice_pitch,
                      num_events_in_wait_list,
                      event_wait_list,
                      event,
                      errcode_ret);

    test_icd_stub_log("Value returned: %p\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueUnmapMemObject(cl_command_queue  command_queue ,
                        cl_mem            memobj ,
                        void *            mapped_ptr ,
                        cl_uint           num_events_in_wait_list ,
                        const cl_event *   event_wait_list ,
                        cl_event *         event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueUnmapMemObject(%p, %p, %p, %u, %p, %p)\n",
                      command_queue,
                      memobj,
                      mapped_ptr,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMigrateMemObjects(cl_command_queue        command_queue ,
                           cl_uint                 num_mem_objects ,
                           const cl_mem *          mem_objects ,
                           cl_mem_migration_flags  flags ,
                           cl_uint                 num_events_in_wait_list ,
                           const cl_event *        event_wait_list ,
                           cl_event *              event) CL_API_SUFFIX__VERSION_1_2
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueMigrateMemObjects(%p, %u, %p, %x, %u, %p, %p)\n",
                      command_queue,
                      num_mem_objects,
                      mem_objects,
                      flags,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}


CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNDRangeKernel(cl_command_queue  command_queue ,
                       cl_kernel         kernel ,
                       cl_uint           work_dim ,
                       const size_t *    global_work_offset ,
                       const size_t *    global_work_size ,
                       const size_t *    local_work_size ,
                       cl_uint           num_events_in_wait_list ,
                       const cl_event *  event_wait_list ,
                       cl_event *        event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueNDRangeKernel(%p, %p, %u, %p, %p, %p, %u, %p, %p)\n",
                      command_queue,
                      kernel,
                      work_dim,
                      global_work_offset,
                      global_work_size,
                      local_work_size,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueTask(cl_command_queue   command_queue ,
              cl_kernel          kernel ,
              cl_uint            num_events_in_wait_list ,
              const cl_event *   event_wait_list ,
              cl_event *         event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueTask(%p, %p, %u, %p, %p)\n",
                      command_queue,
                      kernel,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNativeKernel(cl_command_queue   command_queue ,
                      void (CL_CALLBACK *user_func)(void *),
                      void *             args ,
                      size_t             cb_args ,
                      cl_uint            num_mem_objects ,
                      const cl_mem *     mem_list ,
                      const void **      args_mem_loc ,
                      cl_uint            num_events_in_wait_list ,
                      const cl_event *   event_wait_list ,
                      cl_event *         event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueNativeKernel(%p, %p, %p, %u, %u, %p, %p, %u, %p, %p)\n",
                      command_queue,
                      user_func,
                      args,
                      cb_args,
                      num_mem_objects,
                      mem_list,
                      args_mem_loc,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY void * CL_API_CALL
clGetExtensionFunctionAddressForPlatform(cl_platform_id  platform ,
                                         const char *    func_name) CL_API_SUFFIX__VERSION_1_2
{
    void *return_value = (void *) malloc(sizeof(void *));
    test_icd_stub_log("clGetExtensionFunctionAddressForPlatform(%p, %p)\n",
                      platform,
                      func_name);

    test_icd_stub_log("Value returned: %p\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMarkerWithWaitList(cl_command_queue  command_queue ,
                            cl_uint            num_events_in_wait_list ,
                            const cl_event *   event_wait_list ,
                            cl_event *         event) CL_API_SUFFIX__VERSION_1_2

{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueMarkerWithWaitList(%p, %u, %p, %p)\n",
                      command_queue,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueBarrierWithWaitList(cl_command_queue  command_queue ,
                             cl_uint            num_events_in_wait_list ,
                             const cl_event *   event_wait_list ,
                             cl_event *         event) CL_API_SUFFIX__VERSION_1_2
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueBarrierWithWaitList(%p, %u, %p, %p)\n",
                      command_queue,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clSetPrintfCallback(cl_context           context ,
                    void (CL_CALLBACK *  pfn_notify)(cl_context  program ,
                                                          cl_uint printf_data_len ,
                                                          char *  printf_data_ptr ,
                                                          void *  user_data),
                    void *               user_data) CL_API_SUFFIX__VERSION_1_2
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clSetPrintfCallback(%p, %p, %p)\n",
                      context,
                      pfn_notify,
                      user_data);
    pfn_notify(context, 0, NULL, NULL);
    test_icd_stub_log("setprintf_callback(%p, %u, %p, %p)\n",
                      context,
                      0,
                      NULL,
                      NULL);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}


CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMarker(cl_command_queue     command_queue ,
                cl_event *           event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueMarker(%p, %p)\n", command_queue, event);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWaitForEvents(cl_command_queue  command_queue ,
                       cl_uint           num_events ,
                       const cl_event *  event_list) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueWaitForEvents(%p, %u, %p)\n",
                      command_queue,
                      num_events,
                      event_list);

    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueBarrier(cl_command_queue  command_queue) CL_API_SUFFIX__VERSION_1_0
{
    cl_int return_value = CL_OUT_OF_RESOURCES;
    test_icd_stub_log("clEnqueueBarrier(%p)\n", command_queue);
    test_icd_stub_log("Value returned: %d\n", return_value);
    return return_value;
}

extern cl_int cliIcdDispatchTableCreate(CLIicdDispatchTable **outDispatchTable);

CL_API_ENTRY cl_int CL_API_CALL
clIcdGetPlatformIDsKHR(cl_uint           num_entries,
                       cl_platform_id * platforms,
                       cl_uint *        num_platforms)
{
    cl_int result = CL_SUCCESS;
    if (!initialized) {
        result = cliIcdDispatchTableCreate(&dispatchTable);
        platform = (cl_platform_id) malloc(sizeof(struct _cl_platform_id));
        memset(platform, 0, sizeof(struct _cl_platform_id));

        platform->dispatch = dispatchTable;
        platform->version = "OpenCL 1.2 Stub";
        platform->vendor = "stubvendorxxx";
        platform->profile = "stubprofilexxx";
        platform->name = "ICD_LOADER_TEST_OPENCL_STUB";
        platform->extensions = "cl_khr_icd cl_khr_gl cl_khr_d3d10";
        platform->suffix = "ilts";
        platform->dispatch = dispatchTable;
        initialized = CL_TRUE;
    }

    if ((platforms && num_entries >1) ||
        (platforms && num_entries <= 0) ||
        (!platforms && num_entries >= 1)) {
        result = CL_INVALID_VALUE;
        goto Done;
    }

    if (platforms && num_entries == 1) {
        platforms[0] = platform;
    }

Done:
    if (num_platforms) {
        *num_platforms = 1;
    }

    return result;
}

