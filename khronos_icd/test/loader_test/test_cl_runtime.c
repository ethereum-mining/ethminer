#include <CL/cl.h>
#include "param_struct.h"
#include <platform/icd_test_log.h>

extern cl_command_queue command_queue;

cl_int ret_val;

const struct clRetainCommandQueue_st clRetainCommandQueueData[NUM_ITEMS_clRetainCommandQueue] = {
	{NULL}
};

const struct clGetCommandQueueInfo_st clGetCommandQueueInfoData[NUM_ITEMS_clGetCommandQueueInfo] = {
	{NULL, 0, 0, NULL, NULL}
};

int test_clRetainCommandQueue(const struct clRetainCommandQueue_st *data)
{
    test_icd_app_log("clRetainCommandQueue(%p)\n", command_queue);

    ret_val = clRetainCommandQueue(command_queue);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clGetCommandQueueInfo(const struct clGetCommandQueueInfo_st *data)
{
    test_icd_app_log("clGetCommandQueueInfo(%p, %u, %u, %p, %p)\n",
                     command_queue,
                     data->param_name,
                     data->param_value_size,
                     data->param_value,
                     data->param_value_size_ret);

    ret_val = clGetCommandQueueInfo(command_queue,
                                    data->param_name,
                                    data->param_value_size,
                                    data->param_value,
                                    data->param_value_size_ret);
    
    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_cl_runtime()
{
	int i;

	for (i=0; i<NUM_ITEMS_clRetainCommandQueue; i++)	{
		test_clRetainCommandQueue(&clRetainCommandQueueData[i]);
	}
    
    for (i=0; i<NUM_ITEMS_clGetCommandQueueInfo; i++)    {
        test_clGetCommandQueueInfo(&clGetCommandQueueInfoData[i]);
    }

	return 0;

}
