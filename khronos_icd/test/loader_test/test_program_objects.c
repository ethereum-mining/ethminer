#include <CL/cl.h>
#include "param_struct.h"
#include <platform/icd_test_log.h>

extern cl_context context;
extern cl_program program;
extern cl_platform_id platform;
extern cl_device_id devices;

int ret_val;

extern void CL_CALLBACK program_callback(cl_program _a, void* _b);

const struct clRetainProgram_st clRetainProgramData[NUM_ITEMS_clRetainProgram]=
{
    {NULL}
};

const struct clBuildProgram_st clBuildProgramData[NUM_ITEMS_clBuildProgram]=
{
    {NULL,0,NULL,NULL,program_callback,NULL}
};

const struct clCompileProgram_st clCompileProgramData[NUM_ITEMS_clCompileProgram]=
{
    {NULL,0,NULL,NULL,0,NULL,NULL,program_callback,NULL}
};

const struct clLinkProgram_st clLinkProgramData[NUM_ITEMS_clLinkProgram]=
{
    {NULL,0,NULL,NULL,0,NULL,program_callback,NULL,NULL}
};

const struct clUnloadPlatformCompiler_st clUnloadPlatformCompilerData[NUM_ITEMS_clUnloadPlatformCompiler]=
{
    {NULL}
};

const struct clGetExtensionFunctionAddressForPlatform_st clGetExtensionFunctionAddressForPlatformData[NUM_ITEMS_clGetExtensionFunctionAddressForPlatform]=
{
    {NULL, ""}
};

const struct clGetProgramInfo_st clGetProgramInfoData[NUM_ITEMS_clGetProgramInfo]=
{
    {NULL,0,0,NULL,NULL}
};

const struct clGetProgramBuildInfo_st clGetProgramBuildInfoData[NUM_ITEMS_clGetProgramBuildInfo]=
{
    {NULL,NULL,0,0,NULL,NULL}
};

int test_clRetainProgram(const struct clRetainProgram_st *data)
{
    test_icd_app_log("clRetainProgram(%p)\n",
                    program);

    ret_val=clRetainProgram(program);

    test_icd_app_log("Value returned: %d\n",
                    ret_val);

    return 0;

}

int test_clBuildProgram(const struct clBuildProgram_st *data)
{
    test_icd_app_log("clBuildProgram(%p, %u, %p, %p, %p, %p)\n",
                     program,
                     data->num_devices,
                     &devices,
                     data->options,
                     data->pfn_notify,
                     data->user_data);

    ret_val=clBuildProgram(program,
                        data->num_devices,
                        &devices,
                        data->options,
                        data->pfn_notify,
                        data->user_data);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clCompileProgram(const struct clCompileProgram_st *data)
{
    test_icd_app_log("clCompileProgram(%p, %u, %p, %p, %u, %p, %p, %p)\n",
                     program,
                     data->num_devices,
                     &devices,
                     data->options,
                     data->num_input_headers,
                     data->header_include_names,
                     data->pfn_notify,
                     data->user_data);

    ret_val=clCompileProgram(program,
                            data->num_devices,
                            &devices,
                            data->options,
                            data->num_input_headers,
                            data->headers,
                            data->header_include_names,
                            data->pfn_notify,
                            data->user_data);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clLinkProgram(const struct clLinkProgram_st *data)
{
    cl_program program;
    test_icd_app_log("clLinkProgram(%p, %u, %p, %p, %u, %p, %p, %p, %p)\n",
                     context,
                     data->num_devices,
                     data->device_list,
                     data->options,
                     data->num_input_programs,
                     data->input_programs,
                     data->pfn_notify,
                     data->user_data,
                     data->errcode_ret);

    program=clLinkProgram(context,
                        data->num_devices,
                        data->device_list,
                        data->options,
                        data->num_input_programs,
                        data->input_programs,
                        data->pfn_notify,
                        data->user_data,
                        data->errcode_ret);

    test_icd_app_log("Value returned: %p\n", program);

    return 0;

}

int test_clUnloadPlatformCompiler(const struct clUnloadPlatformCompiler_st *data)
{
    test_icd_app_log("clUnloadPlatformCompiler(%p)\n", platform);

    ret_val=clUnloadPlatformCompiler(platform);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clGetExtensionFunctionAddressForPlatform(const struct clGetExtensionFunctionAddressForPlatform_st *data)
{
    void *return_value;
    test_icd_app_log("clGetExtensionFunctionAddressForPlatform(%p, %p)\n",
                     platform,  
                     data->func_name);

    return_value=clGetExtensionFunctionAddressForPlatform(platform,
                                                        data->func_name);

    test_icd_app_log("Value returned: %p\n", return_value);

    return 0;

}

int test_clGetProgramInfo(const struct clGetProgramInfo_st *data)
{
    test_icd_app_log("clGetProgramInfo(%p, %u, %u, %p, %p)\n",
                     program,
                     data->param_name,
                     data->param_value_size,
                     data->param_value,
                     data->param_value_size_ret);

    ret_val=clGetProgramInfo(program,
                            data->param_name,
                            data->param_value_size,
                            data->param_value,
                            data->param_value_size_ret);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clGetProgramBuildInfo(const struct clGetProgramBuildInfo_st *data)
{
    test_icd_app_log("clGetProgramBuildInfo(%p, %p, %u, %u, %p, %p)\n",
                     program,
                     data->device,
                     data->param_name,
                     data->param_value_size,
                     data->param_value,
                     data->param_value_size_ret);

    ret_val=clGetProgramBuildInfo(program,
                                data->device,
                                data->param_name,
                                data->param_value_size,
                                data->param_value,
                                data->param_value_size_ret);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_program_objects()
{
    int i;

    for (i=0;i<NUM_ITEMS_clRetainProgram;i++)   {
        test_clRetainProgram(&clRetainProgramData[i]);
    }    

    for (i=0;i<NUM_ITEMS_clBuildProgram;i++)    {
        test_clBuildProgram(&clBuildProgramData[i]);
    }

    for (i=0;i<NUM_ITEMS_clCompileProgram;i++)  {
        test_clCompileProgram(&clCompileProgramData[i]);
    }

    for (i=0;i<NUM_ITEMS_clLinkProgram;i++) {
        test_clLinkProgram(&clLinkProgramData[i]);
    }

    for (i=0;i<NUM_ITEMS_clGetExtensionFunctionAddressForPlatform;i++)  {
        test_clGetExtensionFunctionAddressForPlatform(&clGetExtensionFunctionAddressForPlatformData[i]);
    }

    for (i=0;i<NUM_ITEMS_clUnloadPlatformCompiler;i++)  {
        test_clUnloadPlatformCompiler(&clUnloadPlatformCompilerData[i]);
    }

    for (i=0;i<NUM_ITEMS_clGetProgramInfo;i++)  {
        test_clGetProgramInfo(&clGetProgramInfoData[i]);
    }

    for (i=0;i<NUM_ITEMS_clGetProgramBuildInfo;i++) {
        test_clGetProgramBuildInfo(&clGetProgramBuildInfoData[i]);
    }

    return 0;

}

