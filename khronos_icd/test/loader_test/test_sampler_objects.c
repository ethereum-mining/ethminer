#include <CL/cl.h>
#include "param_struct.h"
#include <platform/icd_test_log.h>

extern cl_sampler  sampler;
int ret_val;

const struct clRetainSampler_st clRetainSamplerData[NUM_ITEMS_clRetainSampler]=
{
    { NULL }
};

const struct clGetSamplerInfo_st clGetSamplerInfoData[NUM_ITEMS_clGetSamplerInfo]=
{
    { NULL, 0, 0, NULL, NULL }
};


int test_clRetainSampler(const struct clRetainSampler_st *data)
{
    test_icd_app_log("clRetainSampler(%p)\n", sampler);

    ret_val=clRetainSampler(sampler);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

int test_clGetSamplerInfo(const struct clGetSamplerInfo_st *data)
{
    test_icd_app_log("clGetSamplerInfo(%p, %u, %u, %p, %p)\n",
                     sampler,
                     data->param_name,
                     data->param_value_size,
                     data->param_value,
                     data->param_value_size_ret);

    ret_val=clGetSamplerInfo(sampler,
            data->param_name,
            data->param_value_size,
            data->param_value,
            data->param_value_size_ret);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

int test_sampler_objects()
{
    int i;

    for (i=0;i<NUM_ITEMS_clRetainSampler;i++)	{
        test_clRetainSampler (&clRetainSamplerData[i]);
    }

    for (i=0;i<NUM_ITEMS_clGetSamplerInfo;i++)    {
        test_clGetSamplerInfo(&clGetSamplerInfoData[i]);
    }

    return 0;
}

