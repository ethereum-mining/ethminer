#ifndef _PARAM_STRUCT_H_
#define _PARAM_STRUCT_H_

#include<CL/cl.h>
#include<CL/cl_gl.h>
#include<CL/cl_gl_ext.h>

#ifdef _WIN32
#include <windows.h> /* Needed for gl.h */
#endif
#include<GL/gl.h>

struct clCreateCommandQueue_st
{
    cl_context context;
    cl_device_id device;
    cl_command_queue_properties properties;
    cl_int *errcode_ret;
};

struct clSetCommandQueueProperty_st
{
    cl_command_queue command_queue;
    cl_command_queue_properties properties; 
    cl_bool enable;
    cl_command_queue_properties *old_properties;
};

struct clGetCommandQueueInfo_st 
{
    cl_command_queue command_queue;
    cl_command_queue_info param_name;
    size_t param_value_size;
    void *param_value;
    size_t *param_value_size_ret;
};

struct clCreateContext_st
{
    const cl_context_properties *properties; 
    cl_uint num_devices;
    const cl_device_id *devices; 
    void (CL_CALLBACK*pfn_notify)(const char *errinfo, const void *private_info, size_t cb, void *user_data);
    void *user_data;
    cl_int *errcode_ret;
};

struct clCreateContextFromType_st
{
    const cl_context_properties *properties;
    cl_device_type device_type; 
    void (CL_CALLBACK *pfn_notify)(const char *errinfo, const void *private_info, size_t cb,void *user_data);
    void *user_data;
    cl_int *errcode_ret;
};

struct clRetainContext_st
{
    cl_context context;
};

struct clReleaseContext_st
{
    cl_context context;
};

struct clGetContextInfo_st
{
    cl_context context;
    cl_context_info param_name; 
    size_t param_value_size;
    void *param_value; 
    size_t *param_value_size_ret;
};

struct clGetPlatformIDs_st 
{
    cl_uint num_entries;
    cl_platform_id *platforms; 
    cl_uint *num_platforms;
};

struct clGetPlatformInfo_st 
{
    cl_platform_id platform;
    cl_platform_info param_name; 
    size_t param_value_size;
    void *param_value;
    size_t *param_value_size_ret;
};

struct clGetDeviceIDs_st 
{
    cl_platform_id platform;
    cl_device_type device_type; 
    cl_uint num_entries;
    cl_device_id *devices; 
    cl_uint *num_devices;
};

struct clRetainCommandQueue_st
{
    cl_command_queue command_queue;
};

struct clReleaseCommandQueue_st 
{
    cl_command_queue command_queue;
};

#define NUM_ITEMS_clCreateCommandQueue 1
#define NUM_ITEMS_clRetainCommandQueue 1
#define NUM_ITEMS_clReleaseCommandQueue 1
#define NUM_ITEMS_clGetCommandQueueInfo 1
#define NUM_ITEMS_clSetCommandQueueProperty 1
#define NUM_ITEMS_clCreateContext 1
#define NUM_ITEMS_clCreateContextFromType 1
#define NUM_ITEMS_clRetainContext 1
#define NUM_ITEMS_clReleaseContext 1
#define NUM_ITEMS_clGetContextInfo 1
#define NUM_ITEMS_clGetPlatformIDs 1
#define NUM_ITEMS_clGetPlatformInfo 1
#define NUM_ITEMS_clGetDeviceIDs 1
#define NUM_ITEMS_clGetDeviceInfo 1
#define NUM_ITEMS_clCreateSubDevices 1
#define NUM_ITEMS_clRetainDevice 1
#define NUM_ITEMS_clReleaseDevice 1

struct clGetDeviceInfo_st 
{
    cl_device_id device;
    cl_device_info param_name; 
    size_t param_value_size;
    void *param_value;
    size_t *param_value_size_ret;
};
    
struct clCreateSubDevices_st 
{
    cl_device_id in_device;
    cl_device_partition_property *properties;
    cl_uint num_entries;
    cl_device_id *out_devices;
    cl_uint *num_devices;
};

struct clRetainDevice_st 
{
    cl_device_id device;
};
    
struct clReleaseDevice_st 
{
    cl_device_id device; 
};


#define NUM_ITEMS_clCreateBuffer 1
#define NUM_ITEMS_clCreateSubBuffer 1
#define NUM_ITEMS_clEnqueueReadBuffer 1
#define NUM_ITEMS_clEnqueueWriteBuffer 1
#define NUM_ITEMS_clEnqueueReadBufferRect 1
#define NUM_ITEMS_clEnqueueWriteBufferRect 1
#define NUM_ITEMS_clEnqueueFillBuffer 1
#define NUM_ITEMS_clEnqueueCopyBuffer 1
#define NUM_ITEMS_clEnqueueCopyBufferRect 1
#define NUM_ITEMS_clEnqueueMapBuffer 1
#define NUM_ITEMS_clRetainMemObject 1
#define NUM_ITEMS_clReleaseMemObject 1
#define NUM_ITEMS_clSetMemObjectDestructorCallback 1
#define NUM_ITEMS_clEnqueueUnmapMemObject 1
#define NUM_ITEMS_clGetMemObjectInfo 1

struct clCreateBuffer_st 
{
    cl_context context;
    cl_mem_flags flags;
    size_t size; 
    void *host_ptr;
    cl_int *errcode_ret;
};
struct clCreateSubBuffer_st 
{
    cl_mem buffer;
    cl_mem_flags flags;
    cl_buffer_create_type buffer_create_type;
    const void *buffer_create_info; 
    cl_int *errcode_ret;
};

struct clEnqueueReadBuffer_st 
{
    cl_command_queue command_queue; 
    cl_mem buffer;
    cl_bool blocking_read;
    size_t offset;
    size_t cb;
    void *ptr;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list;
    cl_event *event;
};

struct clEnqueueWriteBuffer_st 
{
    cl_command_queue command_queue;
    cl_mem buffer;
    cl_bool blocking_write;
    size_t offset;
    size_t cb;
    const void *ptr;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list;
    cl_event *event;
};

struct clEnqueueReadBufferRect_st 
{
    cl_command_queue command_queue;
    cl_mem buffer;
    cl_bool blocking_read;
    const size_t * buffer_offset;
    const size_t * host_offset;
    const size_t * region;
    size_t buffer_row_pitch;
    size_t buffer_slice_pitch;
    size_t host_row_pitch;
    size_t host_slice_pitch;
    void *ptr;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list;
    cl_event *event;
};

struct clEnqueueWriteBufferRect_st 
{
    cl_command_queue command_queue;
    cl_mem buffer;
    cl_bool blocking_write;
    const size_t *buffer_offset;
    const size_t *host_offset;
    const size_t *region;
    size_t buffer_row_pitch;
    size_t buffer_slice_pitch;
    size_t host_row_pitch;
    size_t host_slice_pitch;
    void *ptr;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list;
    cl_event *event;
};

struct clEnqueueFillBuffer_st 
{
    cl_command_queue command_queue;
    cl_mem buffer; 
    const void *pattern; 
    size_t pattern_size; 
    size_t offset; 
    size_t cb; 
    cl_uint num_events_in_wait_list; 
    const cl_event *event_wait_list; 
    cl_event *event;
};

struct clEnqueueCopyBuffer_st 
{
    cl_command_queue command_queue;
    cl_mem src_buffer;
    cl_mem dst_buffer;
    size_t src_offset;
    size_t dst_offset;
    size_t cb;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list;
    cl_event *event;
};

struct clEnqueueCopyBufferRect_st 
{
    cl_command_queue command_queue;
    cl_mem src_buffer;
    cl_mem dst_buffer;
    const size_t *src_origin;
    const size_t *dst_origin;
    const size_t *region;
    size_t src_row_pitch;
    size_t src_slice_pitch;
    size_t dst_row_pitch;
    size_t dst_slice_pitch;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list;
    cl_event *event;
};

struct clEnqueueMapBuffer_st 
{
    cl_command_queue command_queue;
    cl_mem buffer;
    cl_bool blocking_map;
    cl_map_flags map_flags;
    size_t offset;
    size_t cb;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list;
    cl_event *event;
    cl_int *errcode_ret;
};

struct clRetainMemObject_st 
{
    cl_mem memobj;
};

struct clReleaseMemObject_st 
{
    cl_mem memobj;
};

struct clSetMemObjectDestructorCallback_st 
{
    cl_mem memobj;
    void (CL_CALLBACK *pfn_notify)(cl_mem memobj, void *user_data);
    void *user_data;
};

struct clEnqueueUnmapMemObject_st 
{
    cl_command_queue command_queue;
    cl_mem memobj;
    void *mapped_ptr;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list;
    cl_event *event;
};

struct clGetMemObjectInfo_st 
{
    cl_mem memobj;
    cl_mem_info param_name;
    size_t param_value_size;
    void *param_value;
    size_t *param_value_size_ret;
};

#define NUM_ITEMS_clCreateProgramWithSource 1
#define NUM_ITEMS_clCreateProgramWithBinary 1
#define NUM_ITEMS_clCreateProgramWithBuiltInKernels 1
#define NUM_ITEMS_clRetainProgram 1
#define NUM_ITEMS_clReleaseProgram 1
#define NUM_ITEMS_clBuildProgram 1
#define NUM_ITEMS_clCompileProgram 1
#define NUM_ITEMS_clLinkProgram 1
#define NUM_ITEMS_clUnloadPlatformCompiler 1
#define NUM_ITEMS_clGetProgramInfo 1
#define NUM_ITEMS_clGetProgramBuildInfo 1
#define NUM_ITEMS_clUnloadCompiler 1
#define NUM_ITEMS_clGetExtensionFunctionAddress 1
#define NUM_ITEMS_clGetExtensionFunctionAddressForPlatform 1

struct clCreateProgramWithSource_st 
{
    cl_context context;
    cl_uint count;
    const char **strings;
    const size_t *lengths;
    cl_int *errcode_ret;
};

struct clCreateProgramWithBinary_st 
{
    cl_context context;
    cl_uint num_devices;
    const cl_device_id *device_list;
    const size_t *lengths;
    const unsigned char **binaries;
    cl_int *binary_status;
    cl_int *errcode_ret;
};

struct clCreateProgramWithBuiltInKernels_st 
{
    cl_context context;
    cl_uint num_devices;
    const cl_device_id *device_list;
    const char *kernel_names;
    cl_int *errcode_ret;
};

struct clRetainProgram_st 
{
    cl_program program;
};

struct clReleaseProgram_st 
{
    cl_program program;
};

struct clBuildProgram_st 
{
    cl_program program;
    cl_uint num_devices;
    const cl_device_id *device_list;
    const char *options;
    void (CL_CALLBACK*pfn_notify)(cl_program program, void *user_data);
    void *user_data;
};

struct clCompileProgram_st 
{
    cl_program program;
    cl_uint num_devices;
    const cl_device_id *device_list;
    const char *options; 
    cl_uint num_input_headers;
    const cl_program *headers;
    const char **header_include_names;
    void (CL_CALLBACK *pfn_notify)(cl_program program, void * user_data);
    void *user_data;
};

struct clLinkProgram_st 
{
    cl_context context;
    cl_uint num_devices;
    const cl_device_id *device_list;
    const char *options; 
    cl_uint num_input_programs;
    const cl_program *input_programs;
    void (CL_CALLBACK *pfn_notify)(cl_program program, void *user_data);
    void *user_data;
    cl_int *errcode_ret; 
};

struct clUnloadPlatformCompiler_st 
{
    cl_platform_id platform;
};

#if 0
struct clUnloadCompiler_st 
{
    void ;
};
#endif

struct clGetExtensionFunctionAddress_st 
{
    const char *func_name;
};

struct clGetExtensionFunctionAddressForPlatform_st 
{
    cl_platform_id platform;
    const char *func_name;
};

struct clGetProgramInfo_st 
{
    cl_program program;
    cl_program_info param_name;
    size_t param_value_size;
    void *param_value;
    size_t *param_value_size_ret;
};

struct clGetProgramBuildInfo_st 
{
    cl_program program;
    cl_device_id device;
    cl_program_build_info param_name;
    size_t param_value_size;
    void *param_value;
    size_t *param_value_size_ret;
};

#define NUM_ITEMS_clCreateImage2D 1
#define NUM_ITEMS_clCreateImage3D 1
#define NUM_ITEMS_clCreateImage 1
#define NUM_ITEMS_clGetSupportedImageFormats 1
#define NUM_ITEMS_clEnqueueCopyImageToBuffer 1
#define NUM_ITEMS_clEnqueueCopyBufferToImage 1
#define NUM_ITEMS_clEnqueueMapImage 1
#define NUM_ITEMS_clEnqueueReadImage 1
#define NUM_ITEMS_clEnqueueWriteImage 1
#define NUM_ITEMS_clEnqueueFillImage 1
#define NUM_ITEMS_clEnqueueCopyImage 1
#define NUM_ITEMS_clGetMemObjectInfo 1
#define NUM_ITEMS_clGetImageInfo 1

struct clCreateImage_st 
{
    cl_context context;
    cl_mem_flags flags;
    const cl_image_format *image_format;
    const cl_image_desc *image_desc; 
    void *host_ptr;
    cl_int *errcode_ret;
};

struct clCreateImage2D_st 
{
    cl_context context;
    cl_mem_flags flags;
    const cl_image_format *image_format;
    size_t image_width;
    size_t image_height;
    size_t image_row_pitch;
    void *host_ptr;
    cl_int *errcode_ret;
};

struct clCreateImage3D_st 
{
    cl_context context;
    cl_mem_flags flags;
    const cl_image_format *image_format;
    size_t image_width;
    size_t image_height;
    size_t image_depth;
    size_t image_row_pitch;
    size_t image_slice_pitch;
    void *host_ptr;
    cl_int *errcode_ret;
};

struct clGetSupportedImageFormats_st 
{
    cl_context context;
    cl_mem_flags flags;
    cl_mem_object_type image_type;
    cl_uint num_entries;
    cl_image_format *image_formats;
    cl_uint *num_image_formats;
};

struct clEnqueueCopyImageToBuffer_st 
{
    cl_command_queue command_queue;
    cl_mem src_image;
    cl_mem dst_buffer;
    const size_t *src_origin;
    const size_t *region;
    size_t dst_offset;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list;
    cl_event *event;
};

struct clEnqueueCopyBufferToImage_st 
{
    cl_command_queue command_queue;
    cl_mem src_buffer;
    cl_mem dst_image;
    size_t src_offset;
    const size_t *dst_origin;
    const size_t *region;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list;
    cl_event *event;
};

struct clEnqueueMapImage_st 
{
    cl_command_queue command_queue;
    cl_mem image;
    cl_bool blocking_map;
    cl_map_flags map_flags;
    const size_t *origin;
    const size_t *region;
    size_t *image_row_pitch;
    size_t *image_slice_pitch;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list;
    cl_event *event;
    cl_int *errcode_ret;
};

struct clEnqueueReadImage_st 
{
    cl_command_queue command_queue;
    cl_mem image;
    cl_bool blocking_read;
    const size_t *origin;
    const size_t *region;
    size_t row_pitch;
    size_t slice_pitch;
    void *ptr;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list;
    cl_event *event;
};

struct clEnqueueWriteImage_st 
{
    cl_command_queue command_queue;
    cl_mem image;
    cl_bool blocking_write;
    const size_t *origin;
    const size_t *region;
    size_t input_row_pitch;
    size_t input_slice_pitch;
    const void *ptr;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list;
    cl_event *event;
};

struct clEnqueueFillImage_st 
{
    cl_command_queue command_queue;
    cl_mem image; 
    const void *fill_color; 
    const size_t *origin; 
    const size_t *region; 
    cl_uint num_events_in_wait_list; 
    const cl_event *event_wait_list; 
    cl_event *event;
};

struct clEnqueueCopyImage_st 
{
    cl_command_queue command_queue;
    cl_mem src_image;
    cl_mem dst_image;
    const size_t *src_origin;
    const size_t *dst_origin;
    const size_t *region;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list; 
    cl_event *event;
};

#if 0
struct clGetMemObjectInfo_st 
{
    cl_mem memobj;
    cl_mem_info param_name;
    size_t param_value_size;
    void *param_value;
    size_t *param_value_size_ret;
};
#endif

struct clGetImageInfo_st 
{
    cl_mem image;
    cl_image_info param_name;
    size_t param_value_size;
    void *param_value;
    size_t *param_value_size_ret;
};

#define NUM_ITEMS_clCreateSampler 1
#define NUM_ITEMS_clRetainSampler 1
#define NUM_ITEMS_clReleaseSampler 1
#define NUM_ITEMS_clGetSamplerInfo 1

struct clCreateSampler_st 
{
    cl_context context;
    cl_bool normalized_coords;
    cl_addressing_mode addressing_mode;
    cl_filter_mode filter_mode;
    cl_int *errcode_ret;
};

struct clRetainSampler_st 
{
    cl_sampler sampler;
};

struct clReleaseSampler_st 
{
    cl_sampler sampler;
};

struct clGetSamplerInfo_st 
{
    cl_sampler sampler;
    cl_sampler_info param_name;
    size_t param_value_size;
    void *param_value;
    size_t *param_value_size_ret;
};

#define NUM_ITEMS_clCreateKernel 1
#define NUM_ITEMS_clCreateKernelsInProgram 1
#define NUM_ITEMS_clRetainKernel 1
#define NUM_ITEMS_clReleaseKernel 1

struct clCreateKernel_st
{
    cl_program program;
    const char *kernel_name; 
    cl_int *errcode_ret;
};

struct clCreateKernelsInProgram_st
{
    cl_program program;
    cl_uint num_kernels;
    cl_kernel *kernels;
    cl_uint *num_kernels_ret;
};

struct clRetainKernel_st
{
    cl_kernel kernel;
};

struct clReleaseKernel_st 
{
    cl_kernel kernel;
};

#define NUM_ITEMS_clSetKernelArg 1
#define NUM_ITEMS_clGetKernelInfo 1
#define NUM_ITEMS_clGetKernelArgInfo 1
#define NUM_ITEMS_clGetKernelWorkGroupInfo 1

struct clSetKernelArg_st 
{
    cl_kernel kernel; 
    cl_uint arg_index;
    size_t arg_size; 
    const void *arg_value;
};

struct clGetKernelInfo_st 
{
    cl_kernel kernel;
    cl_kernel_info param_name; 
    size_t param_value_size;
    void *param_value; 
    size_t *param_value_size_ret;
};

struct clGetKernelArgInfo_st 
{
    cl_kernel kernel;
    cl_uint arg_indx;
    cl_kernel_arg_info param_name;
    size_t param_value_size;
    void *param_value;
    size_t *param_value_size_ret;
};

struct clGetKernelWorkGroupInfo_st 
{
    cl_kernel kernel; 
    cl_device_id device;
    cl_kernel_work_group_info param_name;
    size_t param_value_size; 
    void *param_value;
    size_t *param_value_size_ret;
};

#define NUM_ITEMS_clEnqueueMigrateMemObjects 1
#define NUM_ITEMS_clEnqueueNDRangeKernel 1
#define NUM_ITEMS_clEnqueueTask 1
#define NUM_ITEMS_clEnqueueNativeKernel 1

struct clEnqueueMigrateMemObjects_st 
{
    cl_command_queue command_queue;
    size_t num_mem_objects;
    const cl_mem *mem_objects;
    cl_mem_migration_flags flags;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list;
    cl_event *event;
};

struct clEnqueueNDRangeKernel_st 
{
    cl_command_queue command_queue;
    cl_kernel kernel; cl_uint work_dim;
    const size_t *global_work_offset;
    const size_t *global_work_size;
    const size_t *local_work_size;  
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list; 
    cl_event *event;
};

struct clEnqueueTask_st 
{
    cl_command_queue command_queue; 
    cl_kernel kernel; 
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list; 
    cl_event *event;
};

struct clEnqueueNativeKernel_st 
{
    cl_command_queue command_queue; 
    void (CL_CALLBACK *user_func)(void *);
    void *args; 
    size_t cb_args; 
    cl_uint num_mem_objects;
    const cl_mem *mem_list; 
    const void **args_mem_loc;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list; 
    cl_event *event;
};

#define NUM_ITEMS_clCreateUserEvent 1
#define NUM_ITEMS_clSetUserEventStatus 1
#define NUM_ITEMS_clWaitForEvents 1
#define NUM_ITEMS_clGetEventInfo 1
#define NUM_ITEMS_clSetEventCallback 1
#define NUM_ITEMS_clRetainEvent 1
#define NUM_ITEMS_clReleaseEvent 1

struct clCreateUserEvent_st 
{
    cl_context context;
    cl_int *errcode_ret;
};

struct clSetUserEventStatus_st 
{
    cl_event event;
    cl_int execution_status;
};

struct clWaitForEvents_st 
{
    cl_uint num_events;
    const cl_event *event_list;
};

struct clGetEventInfo_st 
{
    cl_event event;
    cl_event_info param_name; 
    size_t param_value_size;
    void *param_value; 
    size_t *param_value_size_ret;
};

struct clSetEventCallback_st 
{
    cl_event event;
    cl_int command_exec_callback_type;
    void (CL_CALLBACK *pfn_event_notify)(cl_event event, cl_int event_command_exec_status,void *user_data);
    void *user_data;
};

struct clRetainEvent_st 
{
    cl_event event;
};

struct clReleaseEvent_st 
{
    cl_event event;
};

#define NUM_ITEMS_clEnqueueMarker 1
#define NUM_ITEMS_clEnqueueWaitForEvents 1
#define NUM_ITEMS_clEnqueueBarrier 1
#define NUM_ITEMS_clEnqueueMarkerWithWaitList 1
#define NUM_ITEMS_clEnqueueBarrierWithWaitList 1

struct clEnqueueMarker_st 
{
    cl_command_queue command_queue;
    cl_event *event;
};

struct clEnqueueWaitForEvents_st 
{
    cl_command_queue command_queue;
    cl_uint num_events; 
    const cl_event *event_list;
};

struct clEnqueueBarrier_st 
{
    cl_command_queue command_queue;
};

struct clEnqueueMarkerWithWaitList_st 
{
    cl_command_queue command_queue;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list;
    cl_event *event;
};

struct clEnqueueBarrierWithWaitList_st 
{
    cl_command_queue command_queue;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list;
    cl_event *event;
};

#define NUM_ITEMS_clGetEventProfilingInfo 1

struct clGetEventProfilingInfo_st 
{
    cl_event event;
    cl_profiling_info param_name;
    size_t param_value_size; 
    void *param_value;
    size_t *param_value_size_ret;
};

#define NUM_ITEMS_clFlush 1
#define NUM_ITEMS_clFinish 1

struct clFlush_st 
{
    cl_command_queue command_queue;
};

struct clFinish_st 
{
    cl_command_queue command_queue;
};

#define NUM_ITEMS_clCreateFromGLBuffer 1
struct clCreateFromGLBuffer_st 
{
    cl_context context;
    cl_mem_flags flags; 
    GLuint bufobj; 
    int *errcode_ret;
};

#define NUM_ITEMS_clCreateFromGLTexture 1
#define NUM_ITEMS_clCreateFromGLTexture2D 1
#define NUM_ITEMS_clCreateFromGLTexture3D 1

struct clCreateFromGLTexture_st 
{
    cl_context context;
    cl_mem_flags flags; 
    GLenum texture_target;
    GLint miplevel; 
    GLuint texture; 
    cl_int *errcode_ret;
};

struct clCreateFromGLTexture2D_st 
{
    cl_context context;
    cl_mem_flags flags; 
    GLenum texture_target;
    GLint miplevel; 
    GLuint texture; 
    cl_int *errcode_ret;
};

struct clCreateFromGLTexture3D_st 
{
    cl_context context;
    cl_mem_flags flags; 
    GLenum texture_target;
    GLint miplevel; 
    GLuint texture; 
    cl_int *errcode_ret;
};

#define NUM_ITEMS_clCreateFromGLRenderbuffer 1

struct clCreateFromGLRenderbuffer_st 
{
    cl_context context; 
    cl_mem_flags flags;
    GLuint renderbuffer; 
    cl_int *errcode_ret;
};
  
    
    // Query Information [9.8.5]
#define NUM_ITEMS_clGetGLObjectInfo 1
#define NUM_ITEMS_clGetGLTextureInfo 1

struct clGetGLObjectInfo_st 
{
    cl_mem memobj;
    cl_gl_object_type *gl_object_type; 
    GLuint *gl_object_name;
};

struct clGetGLTextureInfo_st 
{
    cl_mem memobj;
    cl_gl_texture_info param_name;
    size_t param_value_size; 
    void *param_value;
    size_t *param_value_size_ret;
};

// Share Objects [9.8.6]

#define NUM_ITEMS_clEnqueueAcquireGLObjects 1
#define NUM_ITEMS_clEnqueueReleaseGLObjects 1

struct clEnqueueAcquireGLObjects_st 
{
    cl_command_queue command_queue;
    cl_uint num_objects; 
    const cl_mem *mem_objects;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list; 
    cl_event *event;
};

struct clEnqueueReleaseGLObjects_st 
{
    cl_command_queue command_queue;
    cl_uint num_objects; 
    const cl_mem *mem_objects;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list; 
    cl_event *event;
};

// CL Event Objects > GL Sync Objects [9.9]
#define NUM_ITEMS_clCreateEventFromGLsyncKHR 1

struct clCreateEventFromGLsyncKHR_st 
{
    cl_context context; 
    cl_GLsync sync; 
    cl_int *errcode_ret;
};

// CL Context > GL Context; Sharegroup [9.7]
#define NUM_ITEMS_clGetGLContextInfoKHR 1

struct clGetGLContextInfoKHR_st 
{
    const cl_context_properties *properties;
    cl_gl_context_info param_name;
    size_t param_value_size; 
    void *param_value;
    size_t *param_value_size_ret;
};

#if 0
// OpenCL/Direct3D 10 Sharing APIs [9.10]

#define NUM_ITEMS_clGetDeviceIDsFromD3D10KHR 1
#define NUM_ITEMS_clCreateFromD3D10BufferKHR 1
#define NUM_ITEMS_clCreateFromD3D10Texture2DKHR 1
#define NUM_ITEMS_clCreateFromD3D10Texture3DKHR 1
#define NUM_ITEMS_clEnqueueAcquireD3D10ObjectsKHR 1
#define NUM_ITEMS_clEnqueueReleaseD3D10ObjectsKHR 1

struct clGetDeviceIDsFromD3D10KHR_st 
{
    cl_platform_id platform;
    cl_d3d10_device_source_khr d3d_device_source;
    void *d3d_object; 
    cl_d3d10_device_set_khr d3d_device_set; 
    cl_uint num_entries;
    cl_device_id *devices; cl_uint *num_devices;
};

struct clCreateFromD3D10BufferKHR_st 
{
    cl_context context; 
    cl_mem_flags flags;
    ID3D10Buffer *resource; 
    cl_int *errcode_ret;
};

struct clCreateFromD3D10Texture2DKHR_st 
{
    cl_context context; 
    cl_mem_flags flags;
    ID3D10Texture2D *resource; 
    UINT subresource;
    cl_int *errcode_ret;
};

struct clCreateFromD3D10Texture3DKHR_st 
{
    cl_context context; 
    cl_mem_flags flags;
    ID3D10Texture3D *resource;
    UINT subresource;
    cl_int *errcode_ret;
};

struct clEnqueueAcquireD3D10ObjectsKHR_st 
{
    cl_command_queue command_queue;
    cl_uint num_objects; 
    const cl_mem *mem_objects;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list;
    cl_event *event;};

struct clEnqueueReleaseD3D10ObjectsKHR_st 
{
    cl_command_queue command_queue;
    cl_uint num_objects; 
    const cl_mem *mem_objects;
    cl_uint num_events_in_wait_list;
    const cl_event *event_wait_list;
    cl_event *event;
};
#endif

#endif /* _PARAM_STRUCT_H_ */
