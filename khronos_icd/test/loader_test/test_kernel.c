#ifndef CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#endif

#include <CL/cl.h>
#include "param_struct.h"
#include <platform/icd_test_log.h>

extern cl_kernel kernel;
extern cl_event event;
extern cl_context  context;
extern cl_command_queue command_queue;
extern cl_device_id devices;
int ret_val;
extern void CL_CALLBACK setevent_callback(cl_event _a, cl_int _b, void* _c);
extern void CL_CALLBACK setprintf_callback(cl_context _a, cl_uint _b, char* _c, void* _d );

struct clRetainKernel_st clRetainKernelData[NUM_ITEMS_clRetainKernel] =
{
	{NULL}
};

int test_clRetainKernel(const struct clRetainKernel_st* data)
{
    test_icd_app_log("clRetainKernel(%p)\n", kernel);
		
    ret_val=clRetainKernel(kernel);
        
    test_icd_app_log("Value returned: %d\n", ret_val);
        
    return 0;
}

struct clSetKernelArg_st clSetKernelArgData[NUM_ITEMS_clSetKernelArg] =
{
	{NULL, 0, 0, NULL}
};

int test_clSetKernelArg(const struct clSetKernelArg_st* data)
{
    test_icd_app_log("clSetKernelArg(%p, %u, %u, %p)\n",
                     kernel, 
                     data->arg_index,    
                     data->arg_size,
                     data->arg_value);
		
    ret_val=clSetKernelArg(kernel, 
                           data->arg_index,    
                           data->arg_size,
                           data->arg_value);
        
    test_icd_app_log("Value returned: %d\n", ret_val);
        
    return 0;
}

struct clGetKernelInfo_st clGetKernelInfoData[NUM_ITEMS_clGetKernelInfo] =
{
	{NULL, 0, 0, NULL, NULL}
};

int test_clGetKernelInfo(const struct clGetKernelInfo_st* data)
{
    test_icd_app_log("clGetKernelInfo(%p, %u, %u, %p, %p)\n",
                     kernel,
                     data->param_name, 
                     data->param_value_size,
                     data->param_value, 
                     data->param_value_size_ret);

    ret_val=clGetKernelInfo(kernel,
                                    data->param_name, 
                                    data->param_value_size,
                                    data->param_value, 
                                    data->param_value_size_ret);
                                    
    test_icd_app_log("Value returned: %d\n", ret_val);
        
        return 0;
}

struct clGetKernelArgInfo_st clGetKernelArgInfoData[NUM_ITEMS_clGetKernelArgInfo] =
{
    {NULL, 0, 0, 0, NULL, NULL}
};

int test_clGetKernelArgInfo(const struct clGetKernelArgInfo_st* data)
{
    test_icd_app_log("clGetKernelArgInfo(%p, %u, %u, %u, %p, %p)\n",
                     kernel, 
                     data->arg_indx,
                     data->param_name,
                     data->param_value_size,
                     data->param_value,
                     data->param_value_size_ret);

    ret_val=clGetKernelArgInfo(kernel, 
            data->arg_indx,
            data->param_name,
            data->param_value_size,
            data->param_value,
            data->param_value_size_ret);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

struct clGetKernelWorkGroupInfo_st clGetKernelWorkGroupInfoData[NUM_ITEMS_clGetKernelWorkGroupInfo] =
{
    {NULL, NULL, 0, 0, NULL, NULL}
};

int test_clGetKernelWorkGroupInfo(const struct clGetKernelWorkGroupInfo_st* data)
{
    test_icd_app_log("clGetKernelWorkGroupInfo(%p, %p, %u, %u, %p, %p)\n",
                     kernel, 
                     devices,
                     data->param_name,
                     data->param_value_size,
                     data->param_value,
                     data->param_value_size_ret);

    ret_val=clGetKernelWorkGroupInfo(kernel, 
            devices,
            data->param_name,
            data->param_value_size,
            data->param_value,
            data->param_value_size_ret);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

struct clEnqueueMigrateMemObjects_st clEnqueueMigrateMemObjectsData[NUM_ITEMS_clEnqueueMigrateMemObjects] =
{
    {NULL, 0, NULL, 0x0, 0, NULL, NULL}
};

int test_clEnqueueMigrateMemObjects(const struct clEnqueueMigrateMemObjects_st* data)
{
    test_icd_app_log("clEnqueueMigrateMemObjects(%p, %u, %p, %x, %u, %p, %p)\n",
                     command_queue,
                     data->num_mem_objects, 
                     data->mem_objects,
                     data->flags,
                     data->num_events_in_wait_list,
                     data->event_wait_list,
                     &event);

    ret_val=clEnqueueMigrateMemObjects(command_queue,
            data->num_mem_objects, 
            data->mem_objects,
            data->flags,
            data->num_events_in_wait_list,
            data->event_wait_list,
            &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

struct clEnqueueNDRangeKernel_st clEnqueueNDRangeKernelData[NUM_ITEMS_clEnqueueNDRangeKernel] =
{
    {NULL, NULL, 0, NULL, NULL, NULL, 0, NULL}
};

int test_clEnqueueNDRangeKernel(const struct clEnqueueNDRangeKernel_st* data)
{
    test_icd_app_log("clEnqueueNDRangeKernel(%p, %p, %u, %p, %p, %p, %u, %p, %p)\n",
                     command_queue,
                     kernel, 
                     data->work_dim,
                     data->global_work_offset,
                     data->global_work_size,
                     data->local_work_size,
                     data->num_events_in_wait_list,
                     data->event_wait_list,
                     &event);

    ret_val=clEnqueueNDRangeKernel(command_queue,
            kernel, 
            data->work_dim,
            data->global_work_offset,
            data->global_work_size,
            data->local_work_size,
            data->num_events_in_wait_list,
            data->event_wait_list,
            &event);

    test_icd_app_log("Value returned: %d\n", ret_val);	

    return 0;
}

struct clEnqueueTask_st clEnqueueTaskData[NUM_ITEMS_clEnqueueTask] =
{
    {NULL, NULL, 0, NULL, NULL}
};

int test_clEnqueueTask(const struct clEnqueueTask_st* data)
{
    test_icd_app_log("clEnqueueTask(%p, %p, %u, %p, %p)\n",
                     command_queue, 
                     kernel, 
                     data->num_events_in_wait_list,
                     data->event_wait_list, 
                     &event);

    ret_val=clEnqueueTask(command_queue, 
            kernel, 
            data->num_events_in_wait_list,
            data->event_wait_list, 
            &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}
struct clEnqueueNativeKernel_st clEnqueueNativeKernelData[NUM_ITEMS_clEnqueueNativeKernel] =
{
    {NULL, NULL, NULL, 0, 0, NULL, NULL, 0, NULL, NULL}
};

int test_clEnqueueNativeKernel(const struct clEnqueueNativeKernel_st* data) {
    test_icd_app_log("clEnqueueNativeKernel(%p, %p, %p, %u, %u, %p, %p, %u, %p, %p)\n",
                     command_queue, 
                     data->user_func,
                     data->args, 
                     data->cb_args, 
                     data->num_mem_objects,
                     data->mem_list, 
                     data->args_mem_loc,
                     data->num_events_in_wait_list,
                     data->event_wait_list, 
                     &event);

    ret_val=clEnqueueNativeKernel(command_queue, 
            data->user_func,
            data->args, 
            data->cb_args, 
            data->num_mem_objects,
            data->mem_list, 
            data->args_mem_loc,
            data->num_events_in_wait_list,
            data->event_wait_list, 
            &event);

    test_icd_app_log("Value returned: %d\n", ret_val);		
    return 0;
}

struct clSetUserEventStatus_st clSetUserEventStatusData[NUM_ITEMS_clSetUserEventStatus] =
{
    {NULL, 0}
};

int test_clSetUserEventStatus(const struct clSetUserEventStatus_st* data)
{
    test_icd_app_log("clSetUserEventStatus(%p, %d)\n",
                     event,
                     data->execution_status);

    ret_val=clSetUserEventStatus(event,
            data->execution_status);

    test_icd_app_log("Value returned: %d\n", ret_val);
    return 0;
}

struct clWaitForEvents_st clWaitForEventsData[NUM_ITEMS_clWaitForEvents] =
{
    {1, NULL}
};

int test_clWaitForEvents(const struct clWaitForEvents_st* data)
{
    test_icd_app_log("clWaitForEvents(%u, %p)\n",
                     data->num_events,
                     &event);

    ret_val=clWaitForEvents(data->num_events,
            &event);

    test_icd_app_log("Value returned: %d\n", ret_val);
    return 0;
}

struct clGetEventInfo_st clGetEventInfoData[NUM_ITEMS_clGetEventInfo] =
{
    {NULL, 0, 0, NULL, NULL}
};

int test_clGetEventInfo(const struct clGetEventInfo_st* data){
    test_icd_app_log("clGetEventInfo(%p, %u, %u, %p, %p)\n",
                     event,
                     data->param_name, 
                     data->param_value_size,
                     data->param_value, 
                     data->param_value_size_ret);

    ret_val=clGetEventInfo(event,
            data->param_name, 
            data->param_value_size,
            data->param_value, 
            data->param_value_size_ret);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

struct clSetEventCallback_st clSetEventCallbackData[NUM_ITEMS_clSetEventCallback] =
{
    {NULL, 0, setevent_callback, NULL}
};

int test_clSetEventCallback(const struct clSetEventCallback_st* data)
{
    test_icd_app_log("clSetEventCallback(%p, %d, %p, %p)\n",
                     event,
                     data->command_exec_callback_type,
                     data->pfn_event_notify,
                     data->user_data);

    ret_val=clSetEventCallback(event,
            data->command_exec_callback_type,
            data->pfn_event_notify,
            data->user_data);

    test_icd_app_log("Value returned: %d\n", ret_val);
    return 0;
}

struct clRetainEvent_st clRetainEventData[NUM_ITEMS_clRetainEvent] =
{
    {NULL}
};

int test_clRetainEvent(const struct clRetainEvent_st* data)
{
    test_icd_app_log("clRetainEvent(%p)\n", event);

    ret_val=clRetainEvent(event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

struct clEnqueueMarker_st clEnqueueMarkerData[NUM_ITEMS_clEnqueueMarker] =
{
    {NULL, NULL}
};

int test_clEnqueueMarker(const struct clEnqueueMarker_st* data)
{
    test_icd_app_log("clEnqueueMarker(%p, %p)\n", command_queue, &event);

    ret_val = clEnqueueMarker(command_queue, &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

struct clEnqueueMarkerWithWaitList_st clEnqueueMarkerWithWaitListData[NUM_ITEMS_clEnqueueMarkerWithWaitList] =
{
    {NULL, 0, NULL, NULL}
};

int test_clEnqueueMarkerWithWaitList(const struct clEnqueueMarkerWithWaitList_st* data)
{
    test_icd_app_log("clEnqueueMarkerWithWaitList(%p, %u, %p, %p)\n",
                     command_queue,
                     data->num_events_in_wait_list,
                     data->event_wait_list,
                     &event);

    ret_val=clEnqueueMarkerWithWaitList(command_queue,
            data->num_events_in_wait_list,
            data->event_wait_list,
            &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

struct clEnqueueBarrierWithWaitList_st clEnqueueBarrierWithWaitListData[NUM_ITEMS_clEnqueueBarrierWithWaitList] =
{
    {NULL, 0, NULL, NULL}
};
int test_clEnqueueBarrierWithWaitList(const struct clEnqueueBarrierWithWaitList_st* data)
{
    test_icd_app_log("clEnqueueBarrierWithWaitList(%p, %u, %p, %p)\n",
                     command_queue,
                     data->num_events_in_wait_list,
                     data->event_wait_list,
                     &event);

    ret_val=clEnqueueBarrierWithWaitList(command_queue,
            data->num_events_in_wait_list,
            data->event_wait_list,
            &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

struct clEnqueueWaitForEvents_st clEnqueueWaitForEventsData[NUM_ITEMS_clEnqueueWaitForEvents] =
{
    {NULL, 0, NULL}
};

int test_clEnqueueWaitForEvents(const struct clEnqueueWaitForEvents_st* data)
{
    test_icd_app_log("clEnqueueWaitForEvents(%p, %u, %p)\n",
                     command_queue,
                     data->num_events, 
                     data->event_list);

    ret_val = clEnqueueWaitForEvents(command_queue,
            data->num_events, 
            data->event_list);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

struct clEnqueueBarrier_st clEnqueueBarrierData[NUM_ITEMS_clEnqueueBarrier] =
{
    {NULL}
};

int test_clEnqueueBarrier(const struct clEnqueueBarrier_st* data)
{
    test_icd_app_log("clEnqueueBarrier(%p)\n", command_queue);

    ret_val = clEnqueueBarrier(command_queue);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}
struct clGetEventProfilingInfo_st clGetEventProfilingInfoData[NUM_ITEMS_clGetEventProfilingInfo] =
{
    {NULL, 0, 0, NULL, NULL}
};

int test_clGetEventProfilingInfo(const struct clGetEventProfilingInfo_st* data)
{
    test_icd_app_log("clGetEventProfilingInfo(%p, %u, %u, %p, %p)\n",
                     event,
                     data->param_name,
                     data->param_value_size, 
                     data->param_value,
                     data->param_value_size_ret);

    ret_val=clGetEventProfilingInfo(event,
            data->param_name,
            data->param_value_size, 
            data->param_value,
            data->param_value_size_ret);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

struct clFlush_st clFlushData[NUM_ITEMS_clFlush] =
{
    {NULL}
};

int test_clFlush(const struct clFlush_st* data)
{
    test_icd_app_log("clFlush(%p)\n", command_queue);

    ret_val=clFlush(command_queue);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

struct clFinish_st clFinishData[NUM_ITEMS_clFinish] =
{
    {NULL}
};

int test_clFinish(const struct clFinish_st* data)
{
    test_icd_app_log("clFinish(%p)\n", command_queue);

    ret_val=clFinish(command_queue);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

int test_kernel()
{
    int i;

    for (i=0; i<NUM_ITEMS_clRetainKernel; i++) { 
        test_clRetainKernel(&clRetainKernelData[i]); 
    }	

    for (i=0; i<NUM_ITEMS_clSetKernelArg; i++) { 
        test_clSetKernelArg(&clSetKernelArgData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clGetKernelInfo; i++) { 
        test_clGetKernelInfo(&clGetKernelInfoData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clGetKernelArgInfo; i++) { 
        test_clGetKernelArgInfo(&clGetKernelArgInfoData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clGetKernelWorkGroupInfo; i++) { 
        test_clGetKernelWorkGroupInfo(&clGetKernelWorkGroupInfoData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clEnqueueMigrateMemObjects; i++) { 
        test_clEnqueueMigrateMemObjects(&clEnqueueMigrateMemObjectsData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clEnqueueNDRangeKernel; i++) { 
        test_clEnqueueNDRangeKernel(&clEnqueueNDRangeKernelData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clEnqueueTask; i++) { 
        test_clEnqueueTask(&clEnqueueTaskData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clEnqueueNativeKernel; i++) { 
        test_clEnqueueNativeKernel(&clEnqueueNativeKernelData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clSetUserEventStatus; i++) { 
        test_clSetUserEventStatus(&clSetUserEventStatusData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clWaitForEvents; i++) { 
        test_clWaitForEvents(&clWaitForEventsData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clGetEventInfo; i++) { 
        test_clGetEventInfo(&clGetEventInfoData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clSetEventCallback; i++) { 
        test_clSetEventCallback(&clSetEventCallbackData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clRetainEvent; i++) { 
        test_clRetainEvent(&clRetainEventData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clEnqueueMarker; i++) { 
        test_clEnqueueMarker(&clEnqueueMarkerData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clEnqueueBarrier; i++) { 
        test_clEnqueueBarrier(&clEnqueueBarrierData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clEnqueueMarkerWithWaitList; i++) { 
        test_clEnqueueMarkerWithWaitList(&clEnqueueMarkerWithWaitListData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clEnqueueBarrierWithWaitList; i++) { 
        test_clEnqueueBarrierWithWaitList(&clEnqueueBarrierWithWaitListData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clGetEventProfilingInfo; i++) { 
        test_clGetEventProfilingInfo(&clGetEventProfilingInfoData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clFlush; i++) { 
        test_clFlush(&clFlushData[i]); 
    }

    for (i=0; i<NUM_ITEMS_clFinish; i++) { 
        test_clFinish(&clFinishData[i]); 
    }

    return 0;
}
