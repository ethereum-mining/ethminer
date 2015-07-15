/*
 * Copyright (c) 2012 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software source and associated documentation files (the "Materials"),
 * to use, copy, modify and compile the Materials to create a binary under the
 * following terms and conditions: 
 *
 * 1. The Materials shall NOT be distributed to any third party;
 *
 * 2. The binary may be distributed without restriction, including without
 * limitation the rights to use, copy, merge, publish, distribute, sublicense,
 * and/or sell copies, and to permit persons to whom the binary is furnished to
 * do so;
 *
 * 3. All modifications to the Materials used to create a binary that is
 * distributed to third parties shall be provided to Khronos with an
 * unrestricted license to use for the purposes of implementing bug fixes and
 * enhancements to the Materials;
 *
 * 4. If the binary is used as part of an OpenCL(TM) implementation, whether
 * binary is distributed together with or separately to that implementation,
 * then recipient must become an OpenCL Adopter and follow the published OpenCL
 * conformance process for that implementation, details at:
 * http://www.khronos.org/conformance/;
 *
 * 5. The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE MATERIALS OR THE USE OR OTHER DEALINGS IN
 * THE MATERIALS.
 * 
 * OpenCL is a trademark of Apple Inc. used under license by Khronos.  
 */

#include "icd_dispatch.h"
#include "icd.h"
#include <stdlib.h>
#include <string.h>

// Platform APIs
CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformIDs(cl_uint          num_entries,
                 cl_platform_id * platforms,
                 cl_uint *        num_platforms) CL_API_SUFFIX__VERSION_1_0
{
    KHRicdVendor* vendor = NULL;
    cl_uint i;

    // initialize the platforms (in case they have not been already)
    khrIcdInitialize();

    if (!num_entries && platforms)
    {
        return CL_INVALID_VALUE;
    }
    if (!platforms && !num_platforms)
    {
        return CL_INVALID_VALUE;
    }
    // set num_platforms to 0 and set all platform pointers to NULL
    if (num_platforms) 
    {
        *num_platforms = 0;
    }
    for (i = 0; i < num_entries && platforms; ++i) 
    {
        platforms[i] = NULL;
    }
    // return error if we have no platforms
    if (!khrIcdState.vendors)
    {
        return CL_PLATFORM_NOT_FOUND_KHR;
    }
    // otherwise enumerate all platforms
    for (vendor = khrIcdState.vendors; vendor; vendor = vendor->next)
    {
        if (num_entries && platforms)
        {
            *(platforms++) = vendor->platform;
            --num_entries;
        }
        if (num_platforms)
        {
            ++(*num_platforms);
        }
    }
    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL 
clGetPlatformInfo(cl_platform_id   platform, 
                  cl_platform_info param_name,
                  size_t           param_value_size, 
                  void *           param_value,
                  size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    // initialize the platforms (in case they have not been already)
    khrIcdInitialize();
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);    
    return platform->dispatch->clGetPlatformInfo(
        platform,
        param_name, 
        param_value_size, 
        param_value, 
        param_value_size_ret);
}

// Device APIs
CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDs(cl_platform_id   platform,
               cl_device_type   device_type, 
               cl_uint          num_entries, 
               cl_device_id *   devices, 
               cl_uint *        num_devices) CL_API_SUFFIX__VERSION_1_0
{
    // initialize the platforms (in case they have not been already)
    khrIcdInitialize();
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);   
    return platform->dispatch->clGetDeviceIDs(
        platform,
        device_type, 
        num_entries, 
        devices, 
        num_devices);
}

CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceInfo(
    cl_device_id    device,
    cl_device_info  param_name, 
    size_t          param_value_size, 
    void *          param_value,
    size_t *        param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return device->dispatch->clGetDeviceInfo(
        device,
        param_name, 
        param_value_size, 
        param_value,
        param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clCreateSubDevices(cl_device_id                         in_device,
                   const cl_device_partition_property * properties,
                   cl_uint                              num_entries,
                   cl_device_id *                       out_devices,
                   cl_uint *                            num_devices) CL_API_SUFFIX__VERSION_1_2
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(in_device, CL_INVALID_DEVICE);
    return in_device->dispatch->clCreateSubDevices(
        in_device,
        properties,
        num_entries,
        out_devices,
        num_devices);
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainDevice(cl_device_id device) CL_API_SUFFIX__VERSION_1_2
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return device->dispatch->clRetainDevice(device);
}
    
CL_API_ENTRY cl_int CL_API_CALL
clReleaseDevice(cl_device_id device) CL_API_SUFFIX__VERSION_1_2
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return device->dispatch->clReleaseDevice(device);
}

// Context APIs  
CL_API_ENTRY cl_context CL_API_CALL
clCreateContext(const cl_context_properties * properties,
                cl_uint                 num_devices,
                const cl_device_id *    devices,
                void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
                void *                  user_data,
                cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    // initialize the platforms (in case they have not been already)
    khrIcdInitialize();
    if (!num_devices || !devices) 
    {
        if (errcode_ret) 
        {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return NULL;
    }
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(devices[0], CL_INVALID_DEVICE);
    return devices[0]->dispatch->clCreateContext(
        properties,
        num_devices,
        devices,
        pfn_notify,
        user_data,
        errcode_ret);
}

CL_API_ENTRY cl_context CL_API_CALL
clCreateContextFromType(const cl_context_properties * properties,
                        cl_device_type          device_type,
                        void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
                        void *                  user_data,
                        cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_platform_id platform = NULL;

    // initialize the platforms (in case they have not been already)
    khrIcdInitialize();

    // determine the platform to use from the properties specified
    khrIcdContextPropertiesGetPlatform(properties, &platform);

    // validate the platform handle and dispatch
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(platform, CL_INVALID_PLATFORM);
    return platform->dispatch->clCreateContextFromType(
        properties,
        device_type,
        pfn_notify,
        user_data,
        errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainContext(cl_context context) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return context->dispatch->clRetainContext(context);
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseContext(cl_context context) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return context->dispatch->clReleaseContext(context);
}

CL_API_ENTRY cl_int CL_API_CALL
clGetContextInfo(cl_context         context, 
                 cl_context_info    param_name, 
                 size_t             param_value_size, 
                 void *             param_value, 
                 size_t *           param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return context->dispatch->clGetContextInfo(
        context, 
        param_name, 
        param_value_size, 
        param_value, 
        param_value_size_ret);
}

// Command Queue APIs
CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueue(cl_context                     context, 
                     cl_device_id                   device, 
                     cl_command_queue_properties    properties,
                     cl_int *                       errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateCommandQueue(
        context, 
        device, 
        properties,
        errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainCommandQueue(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clRetainCommandQueue(command_queue);
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseCommandQueue(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clReleaseCommandQueue(command_queue);
}

CL_API_ENTRY cl_int CL_API_CALL
clGetCommandQueueInfo(cl_command_queue      command_queue,
                      cl_command_queue_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clGetCommandQueueInfo(
        command_queue,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

// Memory Object APIs
CL_API_ENTRY cl_mem CL_API_CALL
clCreateBuffer(cl_context   context,
               cl_mem_flags flags,
               size_t       size,
               void *       host_ptr,
               cl_int *     errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateBuffer(
        context,
        flags,
        size,
        host_ptr,
        errcode_ret);
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateImage(cl_context              context,
                            cl_mem_flags            flags,
                            const cl_image_format * image_format,
                            const cl_image_desc *   image_desc,
                            void *                  host_ptr,
                            cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_2
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateImage(
        context,
        flags,
        image_format,
        image_desc,
        host_ptr,
        errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainMemObject(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    return memobj->dispatch->clRetainMemObject(memobj);
}


CL_API_ENTRY cl_int CL_API_CALL
clReleaseMemObject(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    return memobj->dispatch->clReleaseMemObject(memobj);
}

CL_API_ENTRY cl_int CL_API_CALL
clGetSupportedImageFormats(cl_context           context,
                           cl_mem_flags         flags,
                           cl_mem_object_type   image_type,
                           cl_uint              num_entries,
                           cl_image_format *    image_formats,
                           cl_uint *            num_image_formats) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return context->dispatch->clGetSupportedImageFormats(
        context,
        flags,
        image_type,
        num_entries,
        image_formats,
        num_image_formats);
}
                                    
CL_API_ENTRY cl_int CL_API_CALL
clGetMemObjectInfo(cl_mem           memobj,
                   cl_mem_info      param_name, 
                   size_t           param_value_size,
                   void *           param_value,
                   size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    return memobj->dispatch->clGetMemObjectInfo(
        memobj,
        param_name, 
        param_value_size,
        param_value,
        param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clGetImageInfo(cl_mem           image,
               cl_image_info    param_name, 
               size_t           param_value_size,
               void *           param_value,
               size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(image, CL_INVALID_MEM_OBJECT);
    return image->dispatch->clGetImageInfo(
        image,
        param_name, 
        param_value_size,
        param_value,
        param_value_size_ret);
}

// Sampler APIs
CL_API_ENTRY cl_sampler CL_API_CALL
clCreateSampler(cl_context          context,
                cl_bool             normalized_coords, 
                cl_addressing_mode  addressing_mode, 
                cl_filter_mode      filter_mode,
                cl_int *            errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateSampler(
        context,
        normalized_coords, 
        addressing_mode, 
        filter_mode,
        errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainSampler(cl_sampler sampler) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(sampler, CL_INVALID_SAMPLER);
    return sampler->dispatch->clRetainSampler(sampler);
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseSampler(cl_sampler sampler) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(sampler, CL_INVALID_SAMPLER);
    return sampler->dispatch->clReleaseSampler(sampler);
}

CL_API_ENTRY cl_int CL_API_CALL
clGetSamplerInfo(cl_sampler         sampler,
                 cl_sampler_info    param_name,
                 size_t             param_value_size,
                 void *             param_value,
                 size_t *           param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(sampler, CL_INVALID_SAMPLER);
    return sampler->dispatch->clGetSamplerInfo(
        sampler,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
                            
// Program Object APIs
CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithSource(cl_context        context,
                          cl_uint           count,
                          const char **     strings,
                          const size_t *    lengths,
                          cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateProgramWithSource(
        context,
        count,
        strings,
        lengths,
        errcode_ret);
}

CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithBinary(cl_context                     context,
                          cl_uint                        num_devices,
                          const cl_device_id *           device_list,
                          const size_t *                 lengths,
                          const unsigned char **         binaries,
                          cl_int *                       binary_status,
                          cl_int *                       errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateProgramWithBinary(
        context,
        num_devices,
        device_list,
        lengths,
        binaries,
        binary_status,
        errcode_ret);
}

CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithBuiltInKernels(cl_context            context,
                                  cl_uint               num_devices,
                                  const cl_device_id *  device_list,
                                  const char *          kernel_names,
                                  cl_int *              errcode_ret) CL_API_SUFFIX__VERSION_1_2
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateProgramWithBuiltInKernels(
        context,
        num_devices,
        device_list,
        kernel_names,
        errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainProgram(cl_program program) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return program->dispatch->clRetainProgram(program);
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseProgram(cl_program program) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return program->dispatch->clReleaseProgram(program);
}

CL_API_ENTRY cl_int CL_API_CALL
clBuildProgram(cl_program           program,
               cl_uint              num_devices,
               const cl_device_id * device_list,
               const char *         options, 
               void (CL_CALLBACK *pfn_notify)(cl_program program, void * user_data),
               void *               user_data) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return program->dispatch->clBuildProgram(
        program,
        num_devices,
        device_list,
        options, 
        pfn_notify,
        user_data); 
}

CL_API_ENTRY cl_int CL_API_CALL
clCompileProgram(cl_program           program,
                 cl_uint              num_devices,
                 const cl_device_id * device_list,
                 const char *         options, 
                 cl_uint              num_input_headers,
                 const cl_program *   input_headers,
                 const char **        header_include_names,
                 void (CL_CALLBACK *  pfn_notify)(cl_program program, void * user_data),
                 void *               user_data) CL_API_SUFFIX__VERSION_1_2
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return program->dispatch->clCompileProgram(
        program,
        num_devices,
        device_list,
        options, 
        num_input_headers,
        input_headers,
        header_include_names,
        pfn_notify,
        user_data); 
}

CL_API_ENTRY cl_program CL_API_CALL
clLinkProgram(cl_context           context,
              cl_uint              num_devices,
              const cl_device_id * device_list,
              const char *         options,
              cl_uint              num_input_programs,
              const cl_program *   input_programs,
              void (CL_CALLBACK *  pfn_notify)(cl_program program, void * user_data),
              void *               user_data,
              cl_int *             errcode_ret) CL_API_SUFFIX__VERSION_1_2
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clLinkProgram(
        context,
        num_devices,
        device_list,
        options, 
        num_input_programs,
        input_programs,
        pfn_notify,
        user_data,
        errcode_ret); 
}

CL_API_ENTRY cl_int CL_API_CALL
clUnloadPlatformCompiler(cl_platform_id platform) CL_API_SUFFIX__VERSION_1_2
{
    // initialize the platforms (in case they have not been already)
    khrIcdInitialize();
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);    
    return platform->dispatch->clUnloadPlatformCompiler(platform);
}

CL_API_ENTRY cl_int CL_API_CALL
clGetProgramInfo(cl_program         program,
                 cl_program_info    param_name,
                 size_t             param_value_size,
                 void *             param_value,
                 size_t *           param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return program->dispatch->clGetProgramInfo(
        program,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clGetProgramBuildInfo(cl_program            program,
                      cl_device_id          device,
                      cl_program_build_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return program->dispatch->clGetProgramBuildInfo(
        program,
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
                            
// Kernel Object APIs
CL_API_ENTRY cl_kernel CL_API_CALL
clCreateKernel(cl_program      program,
               const char *    kernel_name,
               cl_int *        errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(program, CL_INVALID_PROGRAM);
    return program->dispatch->clCreateKernel(
        program,
        kernel_name,
        errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clCreateKernelsInProgram(cl_program     program,
                         cl_uint        num_kernels,
                         cl_kernel *    kernels,
                         cl_uint *      num_kernels_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return program->dispatch->clCreateKernelsInProgram(
        program,
        num_kernels,
        kernels,
        num_kernels_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainKernel(cl_kernel    kernel) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return kernel->dispatch->clRetainKernel(kernel);
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseKernel(cl_kernel   kernel) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return kernel->dispatch->clReleaseKernel(kernel);
}

CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArg(cl_kernel    kernel,
               cl_uint      arg_index,
               size_t       arg_size,
               const void * arg_value) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return kernel->dispatch->clSetKernelArg(
        kernel,
        arg_index,
        arg_size,
        arg_value);
}

CL_API_ENTRY cl_int CL_API_CALL
clGetKernelInfo(cl_kernel       kernel,
                cl_kernel_info  param_name,
                size_t          param_value_size,
                void *          param_value,
                size_t *        param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return kernel->dispatch->clGetKernelInfo(
        kernel,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clGetKernelArgInfo(cl_kernel       kernel,
                   cl_uint         arg_indx,
                   cl_kernel_arg_info  param_name,
                   size_t          param_value_size,
                   void *          param_value,
                   size_t *        param_value_size_ret) CL_API_SUFFIX__VERSION_1_2
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return kernel->dispatch->clGetKernelArgInfo(
        kernel,
        arg_indx,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clGetKernelWorkGroupInfo(cl_kernel                  kernel,
                         cl_device_id               device,
                         cl_kernel_work_group_info  param_name,
                         size_t                     param_value_size,
                         void *                     param_value,
                         size_t *                   param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return kernel->dispatch->clGetKernelWorkGroupInfo(
        kernel,
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

// Event Object APIs
CL_API_ENTRY cl_int CL_API_CALL
clWaitForEvents(cl_uint             num_events,
                const cl_event *    event_list) CL_API_SUFFIX__VERSION_1_0
{
    if (!num_events || !event_list) 
    {
        return CL_INVALID_VALUE;        
    }
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event_list[0], CL_INVALID_EVENT);
    return event_list[0]->dispatch->clWaitForEvents(
        num_events,
        event_list);
}

CL_API_ENTRY cl_int CL_API_CALL
clGetEventInfo(cl_event         event,
               cl_event_info    param_name,
               size_t           param_value_size,
               void *           param_value,
               size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return event->dispatch->clGetEventInfo(
        event,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
                            
CL_API_ENTRY cl_int CL_API_CALL
clRetainEvent(cl_event event) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return event->dispatch->clRetainEvent(event);
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseEvent(cl_event event) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return event->dispatch->clReleaseEvent(event);
}

// Profiling APIs
CL_API_ENTRY cl_int CL_API_CALL
clGetEventProfilingInfo(cl_event            event,
                        cl_profiling_info   param_name,
                        size_t              param_value_size,
                        void *              param_value,
                        size_t *            param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return event->dispatch->clGetEventProfilingInfo(
        event,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
                                
// Flush and Finish APIs
CL_API_ENTRY cl_int CL_API_CALL
clFlush(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clFlush(command_queue);
}

CL_API_ENTRY cl_int CL_API_CALL
clFinish(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clFinish(command_queue);
}

// Enqueued Commands APIs
CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBuffer(cl_command_queue    command_queue,
                    cl_mem              buffer,
                    cl_bool             blocking_read,
                    size_t              offset,
                    size_t              cb, 
                    void *              ptr,
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueReadBuffer(
        command_queue,
        buffer,
        blocking_read,
        offset,
        cb, 
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
                            
CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBufferRect(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_read,
    const size_t * buffer_origin,
    const size_t * host_origin, 
    const size_t * region,
    size_t buffer_row_pitch,
    size_t buffer_slice_pitch,
    size_t host_row_pitch,
    size_t host_slice_pitch,                        
    void * ptr,
    cl_uint num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event * event) CL_API_SUFFIX__VERSION_1_1
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueReadBufferRect(
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
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBuffer(cl_command_queue   command_queue, 
                     cl_mem             buffer, 
                     cl_bool            blocking_write, 
                     size_t             offset, 
                     size_t             cb, 
                     const void *       ptr, 
                     cl_uint            num_events_in_wait_list, 
                     const cl_event *   event_wait_list, 
                     cl_event *         event) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueWriteBuffer(
        command_queue, 
        buffer, 
        blocking_write, 
        offset, 
        cb, 
        ptr, 
        num_events_in_wait_list, 
        event_wait_list, 
        event);
}
                            
CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBufferRect(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_read,
    const size_t * buffer_origin,
    const size_t * host_origin, 
    const size_t * region,
    size_t buffer_row_pitch,
    size_t buffer_slice_pitch,
    size_t host_row_pitch,
    size_t host_slice_pitch,                        
    const void * ptr,
    cl_uint num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event * event) CL_API_SUFFIX__VERSION_1_1
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueWriteBufferRect(
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
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueFillBuffer(cl_command_queue   command_queue,
                    cl_mem             buffer, 
                    const void *       pattern, 
                    size_t             pattern_size, 
                    size_t             offset, 
                    size_t             cb, 
                    cl_uint            num_events_in_wait_list, 
                    const cl_event *   event_wait_list, 
                    cl_event *         event) CL_API_SUFFIX__VERSION_1_2
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueFillBuffer(
        command_queue, 
        buffer,
        pattern, 
        pattern_size,
        offset,
        cb, 
        num_events_in_wait_list,
        event_wait_list,
        event);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBuffer(cl_command_queue    command_queue, 
                    cl_mem              src_buffer,
                    cl_mem              dst_buffer, 
                    size_t              src_offset,
                    size_t              dst_offset,
                    size_t              cb, 
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueCopyBuffer(
        command_queue, 
        src_buffer,
        dst_buffer, 
        src_offset,
        dst_offset,
        cb, 
        num_events_in_wait_list,
        event_wait_list,
        event);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBufferRect(
    cl_command_queue command_queue, 
    cl_mem src_buffer,
    cl_mem dst_buffer, 
    const size_t * src_origin,
    const size_t * dst_origin,
    const size_t * region, 
    size_t src_row_pitch,
    size_t src_slice_pitch,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    cl_uint num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event * event) CL_API_SUFFIX__VERSION_1_1
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueCopyBufferRect(
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
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadImage(cl_command_queue     command_queue,
                   cl_mem               image,
                   cl_bool              blocking_read, 
                   const size_t *       origin,
                   const size_t *       region,
                   size_t               row_pitch,
                   size_t               slice_pitch, 
                   void *               ptr,
                   cl_uint              num_events_in_wait_list,
                   const cl_event *     event_wait_list,
                   cl_event *           event) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueReadImage(
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
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteImage(cl_command_queue    command_queue,
                    cl_mem              image,
                    cl_bool             blocking_write, 
                    const size_t *      origin,
                    const size_t *      region,
                    size_t              input_row_pitch,
                    size_t              input_slice_pitch, 
                    const void *        ptr,
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueWriteImage(
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
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueFillImage(cl_command_queue   command_queue,
                   cl_mem             image,
                   const void *       fill_color,
                   const size_t       origin[3], 
                   const size_t       region[3],
                   cl_uint            num_events_in_wait_list,
                   const cl_event *   event_wait_list, 
                   cl_event *         event) CL_API_SUFFIX__VERSION_1_2
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueFillImage(
        command_queue,
        image,
        fill_color, 
        origin,
        region, 
        num_events_in_wait_list,
        event_wait_list,
        event);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyImage(cl_command_queue     command_queue,
                   cl_mem               src_image,
                   cl_mem               dst_image, 
                   const size_t *       src_origin,
                   const size_t *       dst_origin,
                   const size_t *       region, 
                   cl_uint              num_events_in_wait_list,
                   const cl_event *     event_wait_list,
                   cl_event *           event) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueCopyImage(
        command_queue,
        src_image,
        dst_image, 
        src_origin,
        dst_origin,
        region, 
        num_events_in_wait_list,
        event_wait_list,
        event);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyImageToBuffer(cl_command_queue command_queue,
                           cl_mem           src_image,
                           cl_mem           dst_buffer, 
                           const size_t *   src_origin,
                           const size_t *   region, 
                           size_t           dst_offset,
                           cl_uint          num_events_in_wait_list,
                           const cl_event * event_wait_list,
                           cl_event *       event) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueCopyImageToBuffer(
        command_queue,
        src_image,
        dst_buffer, 
        src_origin,
        region, 
        dst_offset,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBufferToImage(cl_command_queue command_queue,
                           cl_mem           src_buffer,
                           cl_mem           dst_image, 
                           size_t           src_offset,
                           const size_t *   dst_origin,
                           const size_t *   region, 
                           cl_uint          num_events_in_wait_list,
                           const cl_event * event_wait_list,
                           cl_event *       event) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueCopyBufferToImage(
        command_queue,
        src_buffer,
        dst_image, 
        src_offset,
        dst_origin,
        region, 
        num_events_in_wait_list,
        event_wait_list,
        event);
}

CL_API_ENTRY void * CL_API_CALL
clEnqueueMapBuffer(cl_command_queue command_queue,
                   cl_mem           buffer,
                   cl_bool          blocking_map, 
                   cl_map_flags     map_flags,
                   size_t           offset,
                   size_t           cb,
                   cl_uint          num_events_in_wait_list,
                   const cl_event * event_wait_list,
                   cl_event *       event,
                   cl_int *         errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueMapBuffer(
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
}

CL_API_ENTRY void * CL_API_CALL
clEnqueueMapImage(cl_command_queue  command_queue,
                  cl_mem            image, 
                  cl_bool           blocking_map, 
                  cl_map_flags      map_flags, 
                  const size_t *    origin,
                  const size_t *    region,
                  size_t *          image_row_pitch,
                  size_t *          image_slice_pitch,
                  cl_uint           num_events_in_wait_list,
                  const cl_event *  event_wait_list,
                  cl_event *        event,
                  cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueMapImage(
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
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueUnmapMemObject(cl_command_queue command_queue,
                        cl_mem           memobj,
                        void *           mapped_ptr,
                        cl_uint          num_events_in_wait_list,
                        const cl_event *  event_wait_list,
                        cl_event *        event) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueUnmapMemObject(
        command_queue,
        memobj,
        mapped_ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMigrateMemObjects(cl_command_queue       command_queue,
                           cl_uint                num_mem_objects,
                           const cl_mem *         mem_objects,
                           cl_mem_migration_flags flags,
                           cl_uint                num_events_in_wait_list,
                           const cl_event *       event_wait_list,
                           cl_event *             event) CL_API_SUFFIX__VERSION_1_2
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueMigrateMemObjects(
        command_queue,
        num_mem_objects,
        mem_objects,
        flags,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNDRangeKernel(cl_command_queue command_queue,
                       cl_kernel        kernel,
                       cl_uint          work_dim,
                       const size_t *   global_work_offset,
                       const size_t *   global_work_size,
                       const size_t *   local_work_size,
                       cl_uint          num_events_in_wait_list,
                       const cl_event * event_wait_list,
                       cl_event *       event) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        work_dim,
        global_work_offset,
        global_work_size,
        local_work_size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueTask(cl_command_queue  command_queue,
              cl_kernel         kernel,
              cl_uint           num_events_in_wait_list,
              const cl_event *  event_wait_list,
              cl_event *        event) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueTask(
        command_queue,
        kernel,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNativeKernel(cl_command_queue  command_queue,
                      void (CL_CALLBACK * user_func)(void *), 
                      void *            args,
                      size_t            cb_args, 
                      cl_uint           num_mem_objects,
                      const cl_mem *    mem_list,
                      const void **     args_mem_loc,
                      cl_uint           num_events_in_wait_list,
                      const cl_event *  event_wait_list,
                      cl_event *        event) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueNativeKernel(
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
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMarkerWithWaitList(cl_command_queue  command_queue,
                            cl_uint           num_events_in_wait_list,
                            const cl_event *  event_wait_list,
                            cl_event *        event) CL_API_SUFFIX__VERSION_1_2
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueMarkerWithWaitList(
        command_queue,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueBarrierWithWaitList(cl_command_queue  command_queue,
                             cl_uint           num_events_in_wait_list,
                             const cl_event *  event_wait_list,
                             cl_event *        event) CL_API_SUFFIX__VERSION_1_2
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueBarrierWithWaitList(
        command_queue,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

CL_API_ENTRY void * CL_API_CALL
clGetExtensionFunctionAddressForPlatform(cl_platform_id platform,
                                         const char *   function_name) CL_API_SUFFIX__VERSION_1_2
{
    // make sure the ICD is initialized
    khrIcdInitialize();    

    // return any ICD-aware extensions
    #define CL_COMMON_EXTENSION_ENTRYPOINT_ADD(name) if (!strcmp(function_name, #name) ) return (void *)&name

    // Are these core or ext?  This is unclear, but they appear to be
    // independent from cl_khr_gl_sharing.
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromGLBuffer);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromGLTexture);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromGLTexture2D);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromGLTexture3D);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromGLRenderbuffer);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clGetGLObjectInfo);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clGetGLTextureInfo);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clEnqueueAcquireGLObjects);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clEnqueueReleaseGLObjects);

    // cl_khr_gl_sharing
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clGetGLContextInfoKHR);

    // cl_khr_gl_event
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateEventFromGLsyncKHR);

#if defined(_WIN32)
    // cl_khr_d3d10_sharing
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clGetDeviceIDsFromD3D10KHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromD3D10BufferKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromD3D10Texture2DKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromD3D10Texture3DKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clEnqueueAcquireD3D10ObjectsKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clEnqueueReleaseD3D10ObjectsKHR);
    // cl_khr_d3d11_sharing
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clGetDeviceIDsFromD3D11KHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromD3D11BufferKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromD3D11Texture2DKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromD3D11Texture3DKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clEnqueueAcquireD3D11ObjectsKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clEnqueueReleaseD3D11ObjectsKHR);
    // cl_khr_dx9_media_sharing
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clGetDeviceIDsFromDX9MediaAdapterKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromDX9MediaSurfaceKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clEnqueueAcquireDX9MediaSurfacesKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clEnqueueReleaseDX9MediaSurfacesKHR);
#endif

    // cl_ext_device_fission
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateSubDevicesEXT);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clRetainDeviceEXT);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clReleaseDeviceEXT);

    // fall back to vendor extension detection

    // FIXME Now that we have a platform id here, we need to validate that it isn't NULL, so shouldn't we have an errcode_ret
    // KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(platform, CL_INVALID_PLATFORM);   
    return platform->dispatch->clGetExtensionFunctionAddressForPlatform(
        platform,
        function_name);
}

// Deprecated APIs
CL_API_ENTRY cl_int CL_API_CALL
clSetCommandQueueProperty(cl_command_queue              command_queue,
                          cl_command_queue_properties   properties, 
                          cl_bool                       enable,
                          cl_command_queue_properties * old_properties) CL_EXT_SUFFIX__VERSION_1_0_DEPRECATED
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clSetCommandQueueProperty(
        command_queue,
        properties, 
        enable,
        old_properties);
}
    
CL_API_ENTRY cl_int CL_API_CALL
clCreateSubDevicesEXT(
    cl_device_id in_device,
    const cl_device_partition_property_ext * partition_properties,
    cl_uint num_entries,
    cl_device_id * out_devices,
    cl_uint * num_devices) CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(in_device, CL_INVALID_DEVICE);
        return in_device->dispatch->clCreateSubDevicesEXT(
        in_device,
        partition_properties,
        num_entries,
        out_devices,
        num_devices);
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainDeviceEXT(cl_device_id device) CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return device->dispatch->clRetainDeviceEXT(device);
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseDeviceEXT(cl_device_id device) CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return device->dispatch->clReleaseDeviceEXT(device);
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateImage2D(cl_context              context,
                cl_mem_flags            flags,
                const cl_image_format * image_format,
                size_t                  image_width,
                size_t                  image_height,
                size_t                  image_row_pitch, 
                void *                  host_ptr,
                cl_int *                errcode_ret) CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateImage2D(
        context,
        flags,
        image_format,
        image_width,
        image_height,
        image_row_pitch, 
        host_ptr,
        errcode_ret);
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateImage3D(cl_context              context,
                cl_mem_flags            flags,
                const cl_image_format * image_format,
                size_t                  image_width, 
                size_t                  image_height,
                size_t                  image_depth, 
                size_t                  image_row_pitch, 
                size_t                  image_slice_pitch, 
                void *                  host_ptr,
                cl_int *                errcode_ret) CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateImage3D(
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
}

CL_API_ENTRY cl_int CL_API_CALL
clUnloadCompiler(void) CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED
{
    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMarker(cl_command_queue    command_queue,
                cl_event *          event) CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueMarker(
        command_queue,
        event);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWaitForEvents(cl_command_queue command_queue,
                       cl_uint          num_events,
                       const cl_event * event_list) CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueWaitForEvents(
        command_queue,
        num_events,
        event_list);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueBarrier(cl_command_queue command_queue) CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueBarrier(command_queue);
}

CL_API_ENTRY void * CL_API_CALL
clGetExtensionFunctionAddress(const char *function_name) CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED
{
    size_t function_name_length = strlen(function_name);
    KHRicdVendor* vendor = NULL;

    // make sure the ICD is initialized
    khrIcdInitialize();    

    // return any ICD-aware extensions
    #define CL_COMMON_EXTENSION_ENTRYPOINT_ADD(name) if (!strcmp(function_name, #name) ) return (void *)&name

    // Are these core or ext?  This is unclear, but they appear to be
    // independent from cl_khr_gl_sharing.
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromGLBuffer);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromGLTexture);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromGLTexture2D);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromGLTexture3D);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromGLRenderbuffer);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clGetGLObjectInfo);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clGetGLTextureInfo);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clEnqueueAcquireGLObjects);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clEnqueueReleaseGLObjects);

    // cl_khr_gl_sharing
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clGetGLContextInfoKHR);

    // cl_khr_gl_event
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateEventFromGLsyncKHR);

#if defined(_WIN32)
    // cl_khr_d3d10_sharing
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clGetDeviceIDsFromD3D10KHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromD3D10BufferKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromD3D10Texture2DKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromD3D10Texture3DKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clEnqueueAcquireD3D10ObjectsKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clEnqueueReleaseD3D10ObjectsKHR);
    // cl_khr_d3d11_sharing
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clGetDeviceIDsFromD3D11KHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromD3D11BufferKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromD3D11Texture2DKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromD3D11Texture3DKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clEnqueueAcquireD3D11ObjectsKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clEnqueueReleaseD3D11ObjectsKHR);
    // cl_khr_dx9_media_sharing
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clGetDeviceIDsFromDX9MediaAdapterKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateFromDX9MediaSurfaceKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clEnqueueAcquireDX9MediaSurfacesKHR);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clEnqueueReleaseDX9MediaSurfacesKHR);
#endif

    // cl_ext_device_fission
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clCreateSubDevicesEXT);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clRetainDeviceEXT);
    CL_COMMON_EXTENSION_ENTRYPOINT_ADD(clReleaseDeviceEXT);

    // fall back to vendor extension detection
    for (vendor = khrIcdState.vendors; vendor; vendor = vendor->next)
    {
        size_t vendor_suffix_length = strlen(vendor->suffix);
        if (vendor_suffix_length <= function_name_length && vendor_suffix_length > 0)
        {            
            const char *function_suffix = function_name+function_name_length-vendor_suffix_length;
            if (!strcmp(function_suffix, vendor->suffix) )
            {
                return vendor->clGetExtensionFunctionAddress(function_name);
            }
        }
    }
    return NULL;
}

// GL and other APIs
CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLBuffer(
    cl_context    context,
    cl_mem_flags  flags,
    GLuint        bufobj,
    int *         errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateFromGLBuffer(
        context,
        flags,
        bufobj,
        errcode_ret);
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLTexture(
    cl_context      context,
    cl_mem_flags    flags,
    cl_GLenum       target,
    cl_GLint        miplevel,
    cl_GLuint       texture,
    cl_int *        errcode_ret) CL_API_SUFFIX__VERSION_1_2
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateFromGLTexture(
        context,
        flags,
        target,
        miplevel,
        texture,
        errcode_ret);
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLTexture2D(
    cl_context      context,
    cl_mem_flags    flags,
    GLenum          target,
    GLint           miplevel,
    GLuint          texture,
    cl_int *        errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateFromGLTexture2D(
        context,
        flags,
        target,
        miplevel,
        texture,
        errcode_ret);
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLTexture3D(
    cl_context      context,
    cl_mem_flags    flags,
    GLenum          target,
    GLint           miplevel,
    GLuint          texture,
    cl_int *        errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateFromGLTexture3D(
        context,
        flags,
        target,
        miplevel,
        texture,
        errcode_ret);
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLRenderbuffer(
    cl_context           context,
    cl_mem_flags         flags,
    GLuint               renderbuffer,
    cl_int *             errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateFromGLRenderbuffer(
        context,
        flags,
        renderbuffer,
        errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clGetGLObjectInfo(
    cl_mem               memobj,
    cl_gl_object_type *  gl_object_type,
    GLuint *             gl_object_name) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    return memobj->dispatch->clGetGLObjectInfo(
        memobj,
        gl_object_type,
        gl_object_name);
}
                  
CL_API_ENTRY cl_int CL_API_CALL clGetGLTextureInfo(
    cl_mem               memobj,
    cl_gl_texture_info   param_name,
    size_t               param_value_size,
    void *               param_value,
    size_t *             param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    return memobj->dispatch->clGetGLTextureInfo(
        memobj,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueAcquireGLObjects(
    cl_command_queue     command_queue,
    cl_uint              num_objects,
    const cl_mem *       mem_objects,
    cl_uint              num_events_in_wait_list,
    const cl_event *     event_wait_list,
    cl_event *           event) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueAcquireGLObjects(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReleaseGLObjects(
    cl_command_queue     command_queue,
    cl_uint              num_objects,
    const cl_mem *       mem_objects,
    cl_uint              num_events_in_wait_list,
    const cl_event *     event_wait_list,
    cl_event *           event) CL_API_SUFFIX__VERSION_1_0
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueReleaseGLObjects(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

CL_API_ENTRY cl_int CL_API_CALL clGetGLContextInfoKHR(
    const cl_context_properties *properties,
    cl_gl_context_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_platform_id platform = NULL;

    // initialize the platforms (in case they have not been already)
    khrIcdInitialize();

    // determine the platform to use from the properties specified
    khrIcdContextPropertiesGetPlatform(properties, &platform);

    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);    
    return platform->dispatch->clGetGLContextInfoKHR(
        properties,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

CL_API_ENTRY cl_event CL_API_CALL clCreateEventFromGLsyncKHR(
	cl_context context,
	cl_GLsync sync,
	cl_int * errcode_ret) CL_API_SUFFIX__VERSION_1_1
{
	KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
	return context->dispatch->clCreateEventFromGLsyncKHR(
		context,
		sync,
		errcode_ret);
}

#if defined(_WIN32)
/*
 *
 * cl_d3d10_sharing_khr
 *
 */

CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDsFromD3D10KHR(
    cl_platform_id platform,
    cl_d3d10_device_source_khr d3d_device_source,
    void *d3d_object,
    cl_d3d10_device_set_khr d3d_device_set,
    cl_uint num_entries, 
    cl_device_id *devices, 
    cl_uint *num_devices)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    return platform->dispatch->clGetDeviceIDsFromD3D10KHR(
        platform,
        d3d_device_source,
        d3d_object,
        d3d_device_set,
        num_entries, 
        devices, 
        num_devices);
}
 
CL_API_ENTRY cl_mem CL_API_CALL 
clCreateFromD3D10BufferKHR(
    cl_context context,
    cl_mem_flags flags,
    ID3D10Buffer *resource,
    cl_int *errcode_ret) 
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateFromD3D10BufferKHR(
        context,
        flags,
        resource,
        errcode_ret);
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateFromD3D10Texture2DKHR(
    cl_context        context,
    cl_mem_flags      flags,
    ID3D10Texture2D * resource,
    UINT              subresource,
    cl_int *          errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateFromD3D10Texture2DKHR(
        context,
        flags,
        resource,
        subresource,
        errcode_ret);
}

CL_API_ENTRY cl_mem CL_API_CALL 
clCreateFromD3D10Texture3DKHR(
    cl_context context,
    cl_mem_flags flags,
    ID3D10Texture3D *resource,
    UINT subresource,
    cl_int *errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateFromD3D10Texture3DKHR(
        context,
        flags,
        resource,
        subresource,
        errcode_ret);        
}

CL_API_ENTRY cl_int CL_API_CALL 
clEnqueueAcquireD3D10ObjectsKHR(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem *mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event) 
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueAcquireD3D10ObjectsKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

CL_API_ENTRY cl_int CL_API_CALL 
clEnqueueReleaseD3D10ObjectsKHR(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem *mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event) 
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueReleaseD3D10ObjectsKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);       
}

/*
 *
 * cl_d3d11_sharing_khr
 *
 */

CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDsFromD3D11KHR(
    cl_platform_id             platform,
    cl_d3d11_device_source_khr d3d_device_source,
    void *                     d3d_object,
    cl_d3d11_device_set_khr    d3d_device_set,
    cl_uint                    num_entries,
    cl_device_id *             devices,
    cl_uint *                  num_devices)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    return platform->dispatch->clGetDeviceIDsFromD3D11KHR(
        platform,
        d3d_device_source,
        d3d_object,
        d3d_device_set,
        num_entries,
        devices,
        num_devices);
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateFromD3D11BufferKHR(
    cl_context     context,
    cl_mem_flags   flags,
    ID3D11Buffer * resource,
    cl_int *       errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateFromD3D11BufferKHR(
        context,
        flags,
        resource,
        errcode_ret);
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateFromD3D11Texture2DKHR(
    cl_context        context,
    cl_mem_flags      flags,
    ID3D11Texture2D * resource,
    UINT              subresource,
    cl_int *          errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateFromD3D11Texture2DKHR(
        context,
        flags,
        resource,
        subresource,
        errcode_ret);
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateFromD3D11Texture3DKHR(
    cl_context        context,
    cl_mem_flags      flags,
    ID3D11Texture3D * resource,
    UINT              subresource,
    cl_int *          errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateFromD3D11Texture3DKHR(
        context,
        flags,
        resource,
        subresource,
        errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueAcquireD3D11ObjectsKHR(
    cl_command_queue command_queue,
    cl_uint          num_objects,
    const cl_mem *   mem_objects,
    cl_uint          num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event *       event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueAcquireD3D11ObjectsKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReleaseD3D11ObjectsKHR(
    cl_command_queue command_queue,
    cl_uint          num_objects,
    const cl_mem *   mem_objects,
    cl_uint          num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event *       event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueReleaseD3D11ObjectsKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

/*
 *
 * cl_khr_dx9_media_sharing
 *
 */

CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDsFromDX9MediaAdapterKHR(
    cl_platform_id                  platform,
    cl_uint                         num_media_adapters,
    cl_dx9_media_adapter_type_khr * media_adapters_type,
    void *                          media_adapters[],
    cl_dx9_media_adapter_set_khr    media_adapter_set,
    cl_uint                         num_entries,
    cl_device_id *                  devices,
    cl_uint *                       num_devices)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    return platform->dispatch->clGetDeviceIDsFromDX9MediaAdapterKHR(
        platform,
        num_media_adapters,
		media_adapters_type,
        media_adapters,
        media_adapter_set,
        num_entries,
        devices,
        num_devices);
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateFromDX9MediaSurfaceKHR(
    cl_context                    context,
    cl_mem_flags                  flags,
    cl_dx9_media_adapter_type_khr adapter_type,
    void *                        surface_info,
    cl_uint                       plane,                                                                          
    cl_int *                      errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateFromDX9MediaSurfaceKHR(
        context,
        flags,
        adapter_type,
        surface_info,
        plane,                                                                          
        errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueAcquireDX9MediaSurfacesKHR(
    cl_command_queue command_queue,
    cl_uint          num_objects,
    const cl_mem *   mem_objects,
    cl_uint          num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event *       event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueAcquireDX9MediaSurfacesKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReleaseDX9MediaSurfacesKHR(
    cl_command_queue command_queue,
    cl_uint          num_objects,
    cl_mem *         mem_objects,
    cl_uint          num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event *       event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueReleaseDX9MediaSurfacesKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

#endif

CL_API_ENTRY cl_int CL_API_CALL 
clSetEventCallback(
    cl_event event,
    cl_int command_exec_callback_type,
    void (CL_CALLBACK *pfn_notify)(cl_event, cl_int, void *),
    void *user_data) CL_API_SUFFIX__VERSION_1_1
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return event->dispatch->clSetEventCallback(
        event,
        command_exec_callback_type,
        pfn_notify,
        user_data);
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateSubBuffer(
    cl_mem buffer,
    cl_mem_flags flags,
    cl_buffer_create_type buffer_create_type,
    const void * buffer_create_info,
    cl_int * errcode_ret) CL_API_SUFFIX__VERSION_1_1
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(buffer, CL_INVALID_MEM_OBJECT);
    return buffer->dispatch->clCreateSubBuffer(
        buffer,
        flags,
        buffer_create_type,
        buffer_create_info,
        errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clSetMemObjectDestructorCallback(
    cl_mem memobj, 
    void (CL_CALLBACK * pfn_notify)( cl_mem, void*), 
    void * user_data )             CL_API_SUFFIX__VERSION_1_1
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    return memobj->dispatch->clSetMemObjectDestructorCallback(
        memobj, 
        pfn_notify,
        user_data);
}

CL_API_ENTRY cl_event CL_API_CALL
clCreateUserEvent(
    cl_context context,
    cl_int * errcode_ret) CL_API_SUFFIX__VERSION_1_1
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateUserEvent(
        context,
        errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clSetUserEventStatus(
    cl_event event,
    cl_int execution_status) CL_API_SUFFIX__VERSION_1_1
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return event->dispatch->clSetUserEventStatus(
        event,
        execution_status);
}

