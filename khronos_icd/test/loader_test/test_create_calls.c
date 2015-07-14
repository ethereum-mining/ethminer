#include <string.h>

#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include <CL/cl.h>
#include "param_struct.h"
#include <platform/icd_test_log.h>

extern void CL_CALLBACK createcontext_callback(const char* a, const void* b, size_t c, void* d);

cl_platform_id*  all_platforms;
cl_platform_id platform;
cl_uint num_platforms;
cl_context context;
cl_command_queue command_queue;
cl_mem buffer;
cl_mem subBuffer;
cl_mem image;
cl_sampler sampler;
cl_program program;
cl_kernel kernel;
cl_event event;
cl_device_id devices;
cl_context_properties context_properties[3] = {
    (cl_context_properties)CL_CONTEXT_PLATFORM, 
    0, 
    0,
};

const struct clGetDeviceIDs_st clGetDeviceIDsData[NUM_ITEMS_clGetDeviceIDs] =
{
    {NULL, 0, 1, NULL, NULL}
};

const struct clCreateSampler_st clCreateSamplerData[NUM_ITEMS_clCreateSampler] =
{
    {NULL, 0x0, 0, 0, NULL},
};

const struct clCreateCommandQueue_st clCreateCommandQueueData[NUM_ITEMS_clCreateCommandQueue] =
{ 
    {NULL, NULL, 0, NULL}
};

const struct clCreateContext_st clCreateContextData[NUM_ITEMS_clCreateContext] =
{
    {NULL, 1, NULL, NULL, NULL, NULL}
};

const struct clCreateContextFromType_st clCreateContextFromTypeData[NUM_ITEMS_clCreateContextFromType] =
{
    {NULL, 0, createcontext_callback, NULL, NULL}
};

const struct clCreateBuffer_st clCreateBufferData[NUM_ITEMS_clCreateBuffer] =
{
    {NULL, 0, 0, NULL, NULL}
};

const struct clCreateSubBuffer_st clCreateSubBufferData[NUM_ITEMS_clCreateSubBuffer] =
{
    {NULL, 0, 0, NULL, NULL}
};

const struct clCreateImage_st clCreateImageData[NUM_ITEMS_clCreateImage] =
{
    { NULL, 0x0, NULL, NULL, NULL, NULL}
};

const struct clCreateImage2D_st clCreateImage2DData[NUM_ITEMS_clCreateImage2D] =
{
    { NULL, 0x0, NULL, 0, 0, 0, NULL, NULL}
};

const struct clCreateImage3D_st clCreateImage3DData[NUM_ITEMS_clCreateImage3D] =
{
    { NULL, 0x0, NULL, 0, 0, 0, 0, 0, NULL, NULL }
};


struct clReleaseMemObject_st clReleaseMemObjectData[NUM_ITEMS_clReleaseMemObject] =
{
    {NULL}
};

struct clReleaseMemObject_st clReleaseMemObjectDataImage[NUM_ITEMS_clReleaseMemObject] =
{
    {NULL}
};const struct clCreateProgramWithSource_st clCreateProgramWithSourceData[NUM_ITEMS_clCreateProgramWithSource] =
{
    {NULL, 0, NULL, NULL, NULL}
};

const struct clCreateProgramWithBinary_st clCreateProgramWithBinaryData[NUM_ITEMS_clCreateProgramWithBinary] =
{
    {NULL, 0, NULL, NULL, NULL, NULL, NULL}
};

const struct clCreateProgramWithBuiltInKernels_st clCreateProgramWithBuiltInKernelsData[NUM_ITEMS_clCreateProgramWithBuiltInKernels] =
{
    {NULL, 0, NULL, NULL, NULL}
};

const struct clCreateKernel_st clCreateKernelData[NUM_ITEMS_clCreateKernel] =
{
    {NULL, NULL, NULL}
};

const struct clCreateKernelsInProgram_st clCreateKernelsInProgramData[NUM_ITEMS_clCreateKernelsInProgram] =
{
    {NULL, 0, NULL, NULL}
};

const struct clCreateUserEvent_st clCreateUserEventData[NUM_ITEMS_clCreateUserEvent] =
{
    {NULL, NULL}
};

const struct clGetPlatformIDs_st clGetPlatformIDsData[NUM_ITEMS_clGetPlatformIDs] =
{
    {0, NULL, 0}
};

/*
 * Some log messages cause log mismatches when ICD loader calls a driver
 * function while initializing platforms. The functions clGetPlatform* are most
 * likely to be called at that time. But nothing stops an ICD loader from
 * calling a ICD driver function anytime.
 *
 * FIXME: Figure out a good way to handle this.
 */
#define ENABLE_MISMATCHING_PRINTS 0

int test_clGetPlatformIDs(const struct clGetPlatformIDs_st* data)
{
    cl_int ret_val;
    size_t param_val_ret_size;
    #define PLATFORM_NAME_SIZE 40
    char platform_name[PLATFORM_NAME_SIZE];
    cl_uint i;    

#if ENABLE_MISMATCHING_PRINTS
    test_icd_app_log("clGetPlatformIDs(%u, %p, %p)\n",
                     data->num_entries,
                     &platforms, 
                     &num_platforms);
#endif

    ret_val = clGetPlatformIDs(0,
                            NULL,
                            &num_platforms);
 
    if (ret_val != CL_SUCCESS){
        return -1;
    }
    
    all_platforms = (cl_platform_id *) malloc (num_platforms * sizeof(cl_platform_id));

    ret_val = clGetPlatformIDs(num_platforms,
                            all_platforms, 
                            NULL); 
  
    if (ret_val != CL_SUCCESS){
        return -1;
    }
   
    for (i = 0; i < num_platforms; i++) {
        ret_val = clGetPlatformInfo(all_platforms[i],
                CL_PLATFORM_NAME,
                PLATFORM_NAME_SIZE,
                (void*)platform_name,
                &param_val_ret_size );  

        if (ret_val == CL_SUCCESS ){
            if(!strcmp(platform_name, "ICD_LOADER_TEST_OPENCL_STUB")) {
                platform = all_platforms[i];                
            }
        }
    }

#if ENABLE_MISMATCHING_PRINTS
    test_icd_app_log("Value returned: %d\n", ret_val);
#endif

    return 0;

}

int test_clGetDeviceIDs(const struct clGetDeviceIDs_st* data)
{
    int ret_val;

    test_icd_app_log("clGetDeviceIDs(%p, %x, %u, %p, %p)\n",
                     platform,
                     data->device_type, 
                     data->num_entries,
                     &devices, 
                     data->num_devices); 
 
    ret_val = clGetDeviceIDs(platform,
                           data->device_type, 
                           data->num_entries,
                           &devices, 
                           data->num_devices);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clCreateContext(const struct clCreateContext_st* data)
{
    test_icd_app_log("clCreateContext(%p, %u, %p, %p, %p, %p)\n",
                     data->properties, 
                     data->num_devices,
                     &devices, 
                     &createcontext_callback,
                     data->user_data,
                     data->errcode_ret);

    context = clCreateContext(data->properties, 
                            data->num_devices,
                            &devices,
                            &createcontext_callback,
                            data->user_data,
                            data->errcode_ret);

    test_icd_app_log("Value returned: %p\n", context);

    return 0;

}

int test_clCreateContextFromType(const struct clCreateContextFromType_st* data)
{
    test_icd_app_log("clCreateContextFromType(%p, %x, %p, %p, %p)\n",
                     context_properties, 
                     data->device_type, 
                     data->pfn_notify,
                     data->user_data,
                     data->errcode_ret);

   
    context = clCreateContextFromType(context_properties, 
                                    data->device_type, 
                                    data->pfn_notify,
                                    data->user_data,
                                    data->errcode_ret);

    test_icd_app_log("Value returned: %p\n", context);

    return 0;

}

int test_clCreateCommandQueue(const struct clCreateCommandQueue_st *data)
{
    test_icd_app_log("clCreateCommandQueue(%p, %p, %x, %p)\n",
                     context,
                     devices,
                     data->properties,
                     data->errcode_ret);

    command_queue = clCreateCommandQueue(context,
                                    devices,
                                    data->properties,
                                    data->errcode_ret);

    test_icd_app_log("Value returned: %p\n", command_queue);

    return 0;

}

int test_clCreateBuffer(const struct clCreateBuffer_st *data)
{
    test_icd_app_log("clCreateBuffer(%p, %x, %u, %p, %p)\n",
                     context,
                     data->flags,
                     data->size, 
                     data->host_ptr,
                     data->errcode_ret);

    buffer = clCreateBuffer(context,
                       data->flags,
                       data->size, 
                       data->host_ptr,
                       data->errcode_ret);
    
    clReleaseMemObjectData->memobj = buffer;

    test_icd_app_log("Value returned: %p\n", buffer);

    return 0;

}

int test_clCreateSubBuffer(const struct clCreateSubBuffer_st *data)
{
    test_icd_app_log("clCreateSubBuffer(%p, %x, %u, %p, %p)\n",
                     buffer,
                     data->flags,
                     data->buffer_create_type,
                     data->buffer_create_info,
                     data->errcode_ret);

    subBuffer = clCreateSubBuffer(buffer,
                                data->flags,
                                data->buffer_create_type,
                                data->buffer_create_info,
                                data->errcode_ret);

    clReleaseMemObjectData->memobj = buffer;

    test_icd_app_log("Value returned: %p\n", subBuffer);

    return 0;

}

int test_clCreateImage(const struct clCreateImage_st *data)
{
    test_icd_app_log("clCreateImage(%p, %x, %p, %p, %p, %p)\n",
                     context,
                     data->flags,
                     data->image_format, 
                     data->image_desc,
                     data->host_ptr,
                     data->errcode_ret);

    image = clCreateImage(context,
                        data->flags,
                        data->image_format, 
                        data->image_desc,
                        data->host_ptr,
                        data->errcode_ret);
    
    clReleaseMemObjectDataImage[0].memobj = image;
    test_icd_app_log("Value returned: %p\n", image);

    return 0;

}

int test_clCreateImage2D(const struct clCreateImage2D_st *data)
{
    test_icd_app_log("clCreateImage2D(%p, %x, %p, %u, %u, %u, %p, %p)\n",
                     context,
                     data->flags,
                     data->image_format, 
                     data->image_width,
                     data->image_height,
                     data->image_row_pitch,
                     data->host_ptr,
                     data->errcode_ret);

    image = clCreateImage2D(context,
                    data->flags,
                    data->image_format, 
                    data->image_width,
                    data->image_height,
                    data->image_row_pitch,
                    data->host_ptr,
                    data->errcode_ret);
    
    clReleaseMemObjectDataImage[0].memobj = image;
    test_icd_app_log("Value returned: %p\n", image);
 
    return 0;

}

int test_clCreateImage3D(const struct clCreateImage3D_st *data)
{
    test_icd_app_log("clCreateImage3D(%p, %x, %p, %u, %u, %u, %u, %u, %p, %p)\n",
                     context,
                     data->flags,
                     data->image_format, 
                     data->image_width,
                     data->image_height,
                     data->image_depth,
                     data->image_row_pitch,
                     data->image_slice_pitch,
                     data->host_ptr,
                     data->errcode_ret);

    image = clCreateImage3D(context,
                    data->flags,
                    data->image_format, 
                    data->image_width,
                    data->image_height,
                    data->image_depth,
                    data->image_row_pitch,
                    data->image_slice_pitch,
                    data->host_ptr,
                    data->errcode_ret);
    
    clReleaseMemObjectDataImage[0].memobj = image;
    test_icd_app_log("Value returned: %p\n", image);

    return 0;

}

int test_clCreateSampler(const struct clCreateSampler_st *data)
{
    test_icd_app_log("clCreateSampler(%p, %u, %u, %u, %p)\n",
                     context,
                     data->normalized_coords,
                     data->addressing_mode,
                     data->filter_mode,
                     data->errcode_ret);

    sampler = clCreateSampler(context,
                            data->normalized_coords,
                            data->addressing_mode,
                            data->filter_mode,
                            data->errcode_ret);

    test_icd_app_log("Value returned: %p\n", sampler);

    return 0;

}

int test_clCreateProgramWithSource(const struct clCreateProgramWithSource_st *data)
{
    test_icd_app_log("clCreateProgramWithSource(%p, %u, %p, %p, %p)\n",
                     context,
                     data->count,
                     data->strings,
                     data->lengths,
                     data->errcode_ret);

    program = clCreateProgramWithSource(context,
                                    data->count,
                                    data->strings,
                                    data->lengths,
                                    data->errcode_ret);

    test_icd_app_log("Value returned: %p\n", program);

    return 0;

}

int test_clCreateProgramWithBinary(const struct clCreateProgramWithBinary_st *data)
{
    test_icd_app_log("clCreateProgramWithBinary(%p, %u, %p, %p, %p, %p, %p)\n",
                     context,
                     data->num_devices,
                     &devices,
                     data->lengths,
                     data->binaries,
                     data->binary_status,
                     data->errcode_ret);
        
    program = clCreateProgramWithBinary(context,
                                        data->num_devices,
                                        &devices,
                                        data->lengths,
                                        data->binaries,
                                        data->binary_status,
                                        data->errcode_ret);

    test_icd_app_log("Value returned: %p\n", program);

    return 0;

}

int test_clCreateProgramWithBuiltInKernels(const struct clCreateProgramWithBuiltInKernels_st *data)
{
    test_icd_app_log("clCreateProgramWithBuiltInKernels(%p, %u, %p, %p, %p)\n",
                     context,
                     data->num_devices,
                     &devices,
                     data->kernel_names,
                     data->errcode_ret);

    program = clCreateProgramWithBuiltInKernels(context,
                                            data->num_devices,
                                            &devices,
                                            data->kernel_names,
                                            data->errcode_ret);

    test_icd_app_log("Value returned: %p\n", program);

    return 0;

}

int test_clCreateKernel(const struct clCreateKernel_st* data)
{
    test_icd_app_log("clCreateKernel(%p, %p, %p)\n",
                     program,
                     data->kernel_name,
                     data->errcode_ret);

    kernel = clCreateKernel(program,
                        data->kernel_name,
                        data->errcode_ret);

    test_icd_app_log("Value returned: %p\n", kernel);

    return 0;

}

int test_clCreateKernelsInProgram(const struct clCreateKernelsInProgram_st* data)
{
    int ret_val;
    test_icd_app_log("clCreateKernelsInProgram(%p, %u, %p, %p)\n",
                     program,
                     data->num_kernels,
                     &kernel,
                     data->num_kernels_ret);

    ret_val = clCreateKernelsInProgram(program,
                                    data->num_kernels,
                                    &kernel,
                                    data->num_kernels_ret);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clCreateUserEvent(const struct clCreateUserEvent_st* data)
{
    test_icd_app_log("clCreateUserEvent(%p, %p)\n",
                     context,
                     data->errcode_ret);

    event = clCreateUserEvent(context,
                            data->errcode_ret);

    test_icd_app_log("Value returned: %p\n", event);

    return 0;

}

const struct clReleaseSampler_st clReleaseSamplerData[NUM_ITEMS_clReleaseSampler] =
{
    { NULL }
};

int test_clReleaseSampler(const struct clReleaseSampler_st *data)
{
    int ret_val = CL_OUT_OF_RESOURCES;

    test_icd_app_log("clReleaseSampler(%p)\n", sampler);

    ret_val = clReleaseSampler(sampler);

    test_icd_app_log("Value returned: %d\n", ret_val);
         
    return 0;

}


int test_clReleaseMemObject(const struct clReleaseMemObject_st *data)
{
    int ret_val = -15;
    test_icd_app_log("clReleaseMemObject(%p)\n", data->memobj);

    ret_val = clReleaseMemObject(data->memobj);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

const struct clReleaseEvent_st clReleaseEventData[NUM_ITEMS_clReleaseEvent] =
{
    {NULL}
};

int test_clReleaseEvent(const struct clReleaseEvent_st* data)
{
    int ret_val = CL_OUT_OF_RESOURCES;

    test_icd_app_log("clReleaseEvent(%p)\n", event);

    ret_val = clReleaseEvent(event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

const struct clReleaseKernel_st clReleaseKernelData[NUM_ITEMS_clReleaseKernel] =
{
    {NULL}
};

int test_clReleaseKernel(const struct clReleaseKernel_st* data)
{
    int ret_val = CL_OUT_OF_RESOURCES;   

    test_icd_app_log("clReleaseKernel(%p)\n", kernel);

    ret_val = clReleaseKernel(kernel);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

const struct clReleaseProgram_st clReleaseProgramData[NUM_ITEMS_clReleaseProgram] =
{
    {NULL}
};

int test_clReleaseProgram(const struct clReleaseProgram_st *data)
{
    int ret_val = CL_OUT_OF_RESOURCES;

    test_icd_app_log("clReleaseProgram(%p)\n", program);

    ret_val = clReleaseProgram(program);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

const struct clReleaseCommandQueue_st clReleaseCommandQueueData[NUM_ITEMS_clReleaseCommandQueue] =
{
    {NULL}
};

int test_clReleaseCommandQueue(const struct clReleaseCommandQueue_st *data)
{
    int ret_val = CL_OUT_OF_RESOURCES;

    test_icd_app_log("clReleaseCommandQueue(%p)\n", command_queue);

    ret_val = clReleaseCommandQueue(command_queue);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

const struct clReleaseContext_st clReleaseContextData[NUM_ITEMS_clReleaseContext] =
{
    {NULL}
};

int test_clReleaseContext(const struct clReleaseContext_st* data)
{
    int ret_val = CL_OUT_OF_RESOURCES; 

    test_icd_app_log("clReleaseContext(%p)\n", context);
             
    ret_val = clReleaseContext(context);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

const struct clReleaseDevice_st clReleaseDeviceData[NUM_ITEMS_clReleaseDevice] =
{
    {NULL}
};

int test_clReleaseDevice(const struct clReleaseDevice_st* data)
{
    int ret_val = CL_OUT_OF_RESOURCES;

    test_icd_app_log("clReleaseDevice(%p)\n", devices); 
 
    ret_val = clReleaseDevice(devices); 

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_create_calls()
{
    test_clGetPlatformIDs(clGetPlatformIDsData);

    context_properties[1] = (cl_context_properties) platform;

    test_clGetDeviceIDs(clGetDeviceIDsData);

    test_clCreateContext(clCreateContextData);  

    test_clReleaseContext(clReleaseContextData);

    test_clCreateContextFromType(clCreateContextFromTypeData);

    test_clCreateCommandQueue(clCreateCommandQueueData);

    test_clCreateBuffer(clCreateBufferData);

    test_clCreateSubBuffer(clCreateSubBufferData);

    test_clCreateImage(clCreateImageData);

    test_clReleaseMemObject(clReleaseMemObjectDataImage);

    test_clCreateImage2D(clCreateImage2DData);

    test_clReleaseMemObject(clReleaseMemObjectDataImage);

    test_clCreateImage3D(clCreateImage3DData);

    test_clCreateSampler(clCreateSamplerData);

    test_clCreateProgramWithSource(clCreateProgramWithSourceData);

    test_clReleaseProgram(clReleaseProgramData);

    test_clCreateProgramWithBinary(clCreateProgramWithBinaryData);

    test_clReleaseProgram(clReleaseProgramData);

    test_clCreateProgramWithBuiltInKernels(clCreateProgramWithBuiltInKernelsData);

    test_clCreateKernel(clCreateKernelData);

    test_clCreateKernelsInProgram(clCreateKernelsInProgramData);

    test_clCreateUserEvent(clCreateUserEventData);

    return 0;

}

int test_release_calls()
{
    test_clReleaseSampler(clReleaseSamplerData);

    test_clReleaseMemObject(clReleaseMemObjectData);

    test_clReleaseMemObject(clReleaseMemObjectDataImage);

    test_clReleaseEvent(clReleaseEventData);

    test_clReleaseKernel(clReleaseKernelData);

    test_clReleaseProgram(clReleaseProgramData);

    test_clReleaseCommandQueue(clReleaseCommandQueueData);

    test_clReleaseContext(clReleaseContextData);

    test_clReleaseDevice(clReleaseDeviceData);

    return 0;
}

