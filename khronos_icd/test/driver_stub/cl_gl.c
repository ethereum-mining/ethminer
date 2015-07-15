#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>
#include <platform/icd_test_log.h>

// Need to rename all CL API functions to prevent ICD loader functions calling
// themselves via the dispatch table. Include this before cl headers.
#include "rename_api.h"

#define SIZE_T_MAX (size_t) 0xFFFFFFFFFFFFFFFFULL

CL_API_ENTRY cl_mem CL_API_CALL
clCreateFromGLBuffer(cl_context      context ,
                     cl_mem_flags    flags ,
                     cl_GLuint       bufret_mem ,
                     int *           errcode_ret ) CL_API_SUFFIX__VERSION_1_0
{    
     cl_mem ret_mem = (cl_mem)(SIZE_T_MAX);  
     test_icd_stub_log("clCreateFromGLBuffer(%p, %x, %u, %p)\n",
                       context,
                       flags,
                       bufret_mem, 
                       errcode_ret);
     test_icd_stub_log("Value returned: %p\n", 
                      ret_mem);
     return ret_mem;
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateFromGLTexture(cl_context       context ,
                      cl_mem_flags     flags ,
                      cl_GLenum        target ,
                      cl_GLint         miplevel ,
                      cl_GLuint        texture ,
                      cl_int *         errcode_ret ) CL_API_SUFFIX__VERSION_1_2
{
     cl_mem ret_mem = (cl_mem)(SIZE_T_MAX);  
     test_icd_stub_log("clCreateFromGLTexture(%p, %x, %d, %d, %u, %p)\n",
                       context ,
                       flags ,
                       target ,
                       miplevel ,
                       texture ,
                       errcode_ret );
     test_icd_stub_log("Value returned: %p\n", 
                      ret_mem);
     return ret_mem;
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateFromGLTexture2D(cl_context       context,
                        cl_mem_flags     flags,
                        cl_GLenum        target,
                        cl_GLint         miplevel,
                        cl_GLuint        texture,
                        cl_int *         errcode_ret ) CL_API_SUFFIX__VERSION_1_0
{
     cl_mem ret_mem = (cl_mem)(SIZE_T_MAX);  
     test_icd_stub_log("clCreateFromGLTexture2D(%p, %x, %d, %d, %u, %p)\n",
                        context,
                        flags,
                        target,
                        miplevel,
                        texture,
                        errcode_ret );
     test_icd_stub_log("Value returned: %p\n", 
                      ret_mem);
     return ret_mem;
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateFromGLTexture3D(cl_context       context,
                        cl_mem_flags     flags,
                        cl_GLenum        target,
                        cl_GLint         miplevel,
                        cl_GLuint        texture,
                        cl_int *         errcode_ret ) CL_API_SUFFIX__VERSION_1_0

{
     cl_mem ret_mem = (cl_mem)(SIZE_T_MAX);  
     test_icd_stub_log("clCreateFromGLTexture3D(%p, %x, %d, %d, %u, %p)\n",
                        context,
                        flags,
                        target,
                        miplevel,
                        texture,
                        errcode_ret );
     test_icd_stub_log("Value returned: %p\n", 
                      ret_mem);
     return ret_mem;
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateFromGLRenderbuffer(cl_context    context,
                           cl_mem_flags  flags,
                           cl_GLuint     renderbuffer,
                           cl_int *      errcode_ret ) CL_API_SUFFIX__VERSION_1_0
{
     cl_mem ret_mem = (cl_mem)(SIZE_T_MAX);  
     test_icd_stub_log("clCreateFromGLRenderbuffer(%p, %x, %d, %p)\n",
                       context,
                       flags,
                       renderbuffer,
                       errcode_ret);
     test_icd_stub_log("Value returned: %p\n", 
                      ret_mem);
     return ret_mem;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetGLObjectInfo(cl_mem                 memobj,
                  cl_gl_object_type *    gl_object_type,
                  cl_GLuint *            gl_object_name ) CL_API_SUFFIX__VERSION_1_0
{  
     cl_int ret_val = -5;
     test_icd_stub_log("clGetGLObjectInfo(%p, %p, %p)\n",
                       memobj,
                       gl_object_type,
                       gl_object_name);
     test_icd_stub_log("Value returned: %p\n", 
                      ret_val);
     return ret_val;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetGLTextureInfo(cl_mem                memobj,
                   cl_gl_texture_info    param_name,
                   size_t                param_value_size,
                   void *                param_value,
                   size_t *              param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0
{
     cl_int ret_val = -5;
     test_icd_stub_log("clGetGLTextureInfo(%p, %u, %u, %p, %p)\n",
                       memobj,
                       param_name,
                       param_value_size,
                       param_value,
                       param_value_size_ret );
     test_icd_stub_log("Value returned: %p\n", 
                      ret_val);
     return ret_val;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueAcquireGLObjects(cl_command_queue       command_queue,
                          cl_uint                num_objects,
                          const cl_mem *         mem_objects,
                          cl_uint                num_events_in_wait_list,
                          const cl_event *       event_wait_list,
                          cl_event *             event ) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret_val = -5;
    test_icd_stub_log("clEnqueueAcquireGLObjects(%p, %u, %p, %u, %p, %p)\n",
                      command_queue,
                      num_objects,
                      mem_objects,
                      num_events_in_wait_list,
                      event_wait_list,
                      event);

     test_icd_stub_log("Value returned: %p\n", 
                      ret_val);
     return ret_val;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReleaseGLObjects(cl_command_queue       command_queue,
                          cl_uint                num_objects,
                          const cl_mem *         mem_objects,
                          cl_uint                num_events_in_wait_list,
                          const cl_event *       event_wait_list,
                          cl_event *             event ) CL_API_SUFFIX__VERSION_1_0

{
     cl_int ret_val = -5;
     test_icd_stub_log("clEnqueueReleaseGLObjects(%p, %u, %p, %u, %p, %p)\n",
                        command_queue,
                        num_objects,
                        mem_objects,
                        num_events_in_wait_list,
                        event_wait_list,
                        event); 
     test_icd_stub_log("Value returned: %p\n", 
                      ret_val);
     return ret_val;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetGLContextInfoKHR(const cl_context_properties *  properties,
                      cl_gl_context_info             param_name,
                      size_t                         param_value_size,
                      void *                         param_value,
                      size_t *                       param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret_val = -5;
    test_icd_stub_log("clGetGLContextInfoKHR(%p, %u, %u, %p, %p)\n",
                      properties,
                      param_name,
                      param_value_size,
                      param_value,
                      param_value_size_ret);

     test_icd_stub_log("Value returned: %p\n", 
                      ret_val);
     return ret_val;
}

CL_API_ENTRY cl_event CL_API_CALL
clCreateEventFromGLsyncKHR(cl_context            context ,
                           cl_GLsync             cl_GLsync ,
                           cl_int *              errcode_ret ) CL_EXT_SUFFIX__VERSION_1_1

{
     cl_event ret_event = (cl_event)(SIZE_T_MAX);
     test_icd_stub_log("clCreateEventFromGLsyncKHR(%p, %p, %p)\n",
                        context,
                        cl_GLsync,
                        errcode_ret);
     test_icd_stub_log("Value returned: %p\n", 
                       ret_event);
     return ret_event;
}
