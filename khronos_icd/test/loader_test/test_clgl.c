#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>
#include "param_struct.h"
#include <platform/icd_test_log.h>

extern cl_context context;
extern cl_mem buffer;
extern cl_command_queue command_queue;
extern cl_event event;
extern cl_context_properties context_properties[3];
cl_int ret_val;
cl_mem ret_mem;

struct clCreateFromGLBuffer_st clCreateFromGLBufferData[NUM_ITEMS_clCreateFromGLBuffer] = {
	{NULL, 0x0, 0, NULL}
};

int test_clCreateFromGLBuffer(const struct clCreateFromGLBuffer_st* data)
{

    test_icd_app_log("clCreateFromGLBuffer(%p, %x, %u, %p)\n",
                     context,
                     data->flags,
                     data->bufobj, 
                     data->errcode_ret);

    ret_mem = clCreateFromGLBuffer(context,
                                   data->flags,
                                   data->bufobj, 
                                   data->errcode_ret);	

    test_icd_app_log("Value returned: %p\n", ret_mem);
    
    return 0;
}

struct clCreateFromGLTexture_st clCreateFromGLTextureData[NUM_ITEMS_clCreateFromGLTexture] = {
    {NULL, 0x0, 0, 0, 0, NULL}
};

int test_clCreateFromGLTexture(const struct clCreateFromGLTexture_st* data)
{
    test_icd_app_log("clCreateFromGLTexture(%p, %x, %d, %d, %u, %p)\n",
                     context,
                     data->flags, 
                     data->texture_target,
                     data->miplevel, 
                     data->texture, 
                     data->errcode_ret);

    ret_mem = clCreateFromGLTexture(context,
                                    data->flags, 
                                    data->texture_target,
                                    data->miplevel, 
                                    data->texture, 
                                    data->errcode_ret);

    test_icd_app_log("Value returned: %p\n", ret_mem);
    
    return 0;
}

struct clCreateFromGLTexture2D_st clCreateFromGLTexture2DData[NUM_ITEMS_clCreateFromGLTexture2D] = {
    {NULL, 0x0, 0, 0, 0, NULL}
};

int test_clCreateFromGLTexture2D(const struct clCreateFromGLTexture2D_st* data)
{
    test_icd_app_log("clCreateFromGLTexture2D(%p, %x, %d, %d, %u, %p)\n",
                     context,
                     data->flags, 
                     data->texture_target,
                     data->miplevel, 
                     data->texture, 
                     data->errcode_ret);

    ret_mem = clCreateFromGLTexture2D(context,
                                      data->flags, 
                                      data->texture_target,
                                      data->miplevel, 
                                      data->texture, 
                                      data->errcode_ret);

    test_icd_app_log("Value returned: %p\n", ret_mem);
    
    return 0;
}

struct clCreateFromGLTexture3D_st clCreateFromGLTexture3DData[NUM_ITEMS_clCreateFromGLTexture3D] = {
    {NULL, 0, 0, 0, 0, NULL}
};

int test_clCreateFromGLTexture3D(const struct clCreateFromGLTexture3D_st* data)
{	
    test_icd_app_log("clCreateFromGLTexture3D(%p, %x, %d, %d, %u, %p)\n",
                     context,
                     data->flags,
                     data->texture_target,
                     data->miplevel, 
                     data->texture, 
                     data->errcode_ret);

    ret_mem = clCreateFromGLTexture3D(context,
                                      data->flags, 
                                      data->texture_target,
                                      data->miplevel, 
                                      data->texture, 
                                      data->errcode_ret);	

    test_icd_app_log("Value returned: %p\n", ret_mem);

     return 0;
}

struct clCreateFromGLRenderbuffer_st clCreateFromGLRenderbufferData[NUM_ITEMS_clCreateFromGLRenderbuffer] = {
    {NULL, 0x0, 0, NULL}
};

int test_clCreateFromGLRenderbuffer(const struct clCreateFromGLRenderbuffer_st* data)
{
    test_icd_app_log("clCreateFromGLRenderbuffer(%p, %x, %d, %p)\n",
                     context, 
                     data->flags,
                     data->renderbuffer, 
                     data->errcode_ret);

    ret_mem = clCreateFromGLRenderbuffer(context, 
                                         data->flags,
                                         data->renderbuffer, 
                                         data->errcode_ret);

    test_icd_app_log("Value returned: %p\n", ret_mem);

    return 0;
}

struct clGetGLObjectInfo_st clGetGLObjectInfoData[NUM_ITEMS_clGetGLObjectInfo] = {
    {NULL, NULL, NULL}
};

int test_clGetGLObjectInfo(const struct clGetGLObjectInfo_st* data)
{
    test_icd_app_log("clGetGLObjectInfo(%p, %p, %p)\n",
                     buffer,
                     data->gl_object_type, 
                     data->gl_object_name);

    ret_val = clGetGLObjectInfo(buffer,
                                data->gl_object_type, 
                                data->gl_object_name);

    test_icd_app_log("Value returned: %p\n", ret_val);

}

struct clGetGLTextureInfo_st clGetGLTextureInfoData[NUM_ITEMS_clGetGLTextureInfo] = {
    {NULL, 0, 0, NULL, NULL}
};

int test_clGetGLTextureInfo(const struct clGetGLTextureInfo_st* data)
{
    test_icd_app_log("clGetGLTextureInfo(%p, %u, %u, %p, %p)\n",
                     buffer,
                     data->param_name,
                     data->param_value_size, 
                     data->param_value,
                     data->param_value_size_ret);

    ret_val = clGetGLTextureInfo (buffer,
                                  data->param_name,
                                  data->param_value_size, 
                                  data->param_value,
                                  data->param_value_size_ret);

    test_icd_app_log("Value returned: %p\n", ret_val);

    return 0;
}

struct clEnqueueAcquireGLObjects_st clEnqueueAcquireGLObjectsData[NUM_ITEMS_clEnqueueAcquireGLObjects] = {
    {NULL, 0, NULL, 0, NULL, NULL}
};

int test_clEnqueueAcquireGLObjects(const struct clEnqueueAcquireGLObjects_st* data)
{
    test_icd_app_log("clEnqueueAcquireGLObjects(%p, %u, %p, %u, %p, %p)\n",
                     command_queue,
                     data->num_objects, 
                     data->mem_objects,
                     data->num_events_in_wait_list,
                     &event,
                     &event);

    ret_val = clEnqueueAcquireGLObjects (command_queue,
                                         data->num_objects, 
                                         data->mem_objects,
                                         data->num_events_in_wait_list,
                                         &event, 
                                         &event);

    test_icd_app_log("Value returned: %p\n", ret_val);

    return 0;
}

struct clEnqueueReleaseGLObjects_st clEnqueueReleaseGLObjectsData[NUM_ITEMS_clEnqueueReleaseGLObjects] = {
    {NULL, 0, NULL, 0, NULL, NULL}
};

int test_clEnqueueReleaseGLObjects(const struct clEnqueueReleaseGLObjects_st* data)
{
    test_icd_app_log("clEnqueueReleaseGLObjects(%p, %u, %p, %u, %p, %p)\n",
                     command_queue,
                     data->num_objects, 
                     data->mem_objects,
                     data->num_events_in_wait_list,
                     &event, 
                     &event);

    ret_val = clEnqueueReleaseGLObjects (command_queue,
                                         data->num_objects, 
                                         data->mem_objects,
                                         data->num_events_in_wait_list,
                                         &event, 
                                         &event);


    test_icd_app_log("Value returned: %p\n", ret_val);

    return 0;
}

struct clCreateEventFromGLsyncKHR_st clCreateEventFromGLsyncKHRData[NUM_ITEMS_clCreateEventFromGLsyncKHR] = {
    {NULL, NULL, NULL}
};

typedef CL_API_ENTRY cl_event
(CL_API_CALL *PFN_clCreateEventFromGLsyncKHR)(cl_context           /* context */,
                                              cl_GLsync            /* cl_GLsync */,
                                              cl_int *             /* errcode_ret */);

int test_clCreateEventFromGLsyncKHR(const struct clCreateEventFromGLsyncKHR_st* data)
{   cl_event ret_event;
    PFN_clCreateEventFromGLsyncKHR pfn_clCreateEventFromGLsyncKHR = NULL;

    test_icd_app_log("clCreateEventFromGLsyncKHR(%p, %p, %p)\n",
                     context, 
                     data->sync, 
                     data->errcode_ret);

    pfn_clCreateEventFromGLsyncKHR = clGetExtensionFunctionAddress("clCreateEventFromGLsyncKHR");
    if (!pfn_clCreateEventFromGLsyncKHR) {
        test_icd_app_log("clGetExtensionFunctionAddress failed!\n");
        return 1;
    }

    ret_event = pfn_clCreateEventFromGLsyncKHR (context, 
                                            data->sync, 
                                            data->errcode_ret);

    test_icd_app_log("Value returned: %p\n", ret_event);
    return 0;
}

struct clGetGLContextInfoKHR_st clGetGLContextInfoKHRData[NUM_ITEMS_clGetGLContextInfoKHR] = {
    {NULL, 0, 0, NULL, NULL}
};

typedef CL_API_ENTRY cl_int
(CL_API_CALL *PFN_clGetGLContextInfoKHR)(const cl_context_properties * /* properties */,
                                         cl_gl_context_info            /* param_name */,
                                         size_t                        /* param_value_size */,
                                         void *                        /* param_value */,
                                         size_t *                      /* param_value_size_ret */);
 
int test_clGetGLContextInfoKHR(const struct clGetGLContextInfoKHR_st* data)
{
    PFN_clGetGLContextInfoKHR pfn_clGetGLContextInfoKHR = NULL;
    test_icd_app_log("clGetGLContextInfoKHR(%p, %u, %u, %p, %p)\n",
                     context_properties,
                     data->param_name,
                     data->param_value_size, 
                     data->param_value,
                     data->param_value_size_ret);

    pfn_clGetGLContextInfoKHR = clGetExtensionFunctionAddress("clGetGLContextInfoKHR");
    if (!pfn_clGetGLContextInfoKHR) {
        test_icd_app_log("clGetExtensionFunctionAddress failed!\n");
        return 1;
    }

    ret_val = pfn_clGetGLContextInfoKHR(context_properties,
                                    data->param_name,
                                    data->param_value_size, 
                                    data->param_value,
                                    data->param_value_size_ret);

    test_icd_app_log("Value returned: %p\n", ret_val);
    return 0;

}

int test_OpenGL_share()
{
	int i;
    
    for(i=0;i<NUM_ITEMS_clCreateFromGLBuffer;i++)
		test_clCreateFromGLBuffer(&clCreateFromGLBufferData[i]);

    for(i=0;i<NUM_ITEMS_clCreateFromGLTexture;i++)
		test_clCreateFromGLTexture(&clCreateFromGLTextureData[i]);

    for(i=0;i<NUM_ITEMS_clCreateFromGLTexture2D;i++)
		test_clCreateFromGLTexture2D(&clCreateFromGLTexture2DData[i]);

    for(i=0;i<NUM_ITEMS_clCreateFromGLTexture3D;i++)
		test_clCreateFromGLTexture3D(&clCreateFromGLTexture3DData[i]);

    for(i=0;i<NUM_ITEMS_clCreateFromGLRenderbuffer;i++)
		test_clCreateFromGLRenderbuffer(&clCreateFromGLRenderbufferData[i]);

    for(i=0;i<NUM_ITEMS_clGetGLObjectInfo;i++)
		test_clGetGLObjectInfo(&clGetGLObjectInfoData[i]);

    for(i=0;i<NUM_ITEMS_clGetGLTextureInfo;i++)
		test_clGetGLTextureInfo(&clGetGLTextureInfoData[i]);

    for(i=0;i<NUM_ITEMS_clEnqueueAcquireGLObjects;i++)
		test_clEnqueueAcquireGLObjects(&clEnqueueAcquireGLObjectsData[i]);
        
    for(i=0;i<NUM_ITEMS_clEnqueueReleaseGLObjects;i++)
		test_clEnqueueReleaseGLObjects(&clEnqueueReleaseGLObjectsData[i]);	
    
    for(i=0;i<NUM_ITEMS_clCreateEventFromGLsyncKHR;i++)
		test_clCreateEventFromGLsyncKHR(&clCreateEventFromGLsyncKHRData[i]);
    
    for(i=0;i<NUM_ITEMS_clGetGLContextInfoKHR;i++)
		test_clGetGLContextInfoKHR(&clGetGLContextInfoKHRData[i]);
    
    return 0;
}
