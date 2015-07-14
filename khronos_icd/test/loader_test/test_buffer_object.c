#include <CL/cl.h>
#include "param_struct.h"
#include <platform/icd_test_log.h>

extern cl_mem buffer;
extern cl_command_queue command_queue;
extern cl_event event;

static int ret_val;

extern void CL_CALLBACK setmemobjectdestructor_callback(cl_mem _a, void* _b);

const struct clEnqueueReadBuffer_st clEnqueueReadBufferData[NUM_ITEMS_clEnqueueReadBuffer] =
{
    {NULL, NULL, 0, 0, 0, NULL, 0, NULL, NULL}
};

const struct clEnqueueWriteBuffer_st clEnqueueWriteBufferData[NUM_ITEMS_clEnqueueWriteBuffer] =
{
    {NULL, NULL, 0, 0, 0, NULL, 0, NULL, NULL}
};

const struct clEnqueueReadBufferRect_st clEnqueueReadBufferRectData[NUM_ITEMS_clEnqueueReadBufferRect] =
{
    {NULL, NULL, 0, NULL, NULL, NULL, 0, 0, 0, 0, NULL, 0, NULL, NULL}
};

const struct clEnqueueWriteBufferRect_st clEnqueueWriteBufferRectData[NUM_ITEMS_clEnqueueWriteBufferRect] =
{
    {NULL, NULL, 0, NULL, NULL, NULL, 0, 0, 0, 0, NULL, 0, NULL, NULL}
};

const struct clEnqueueFillBuffer_st clEnqueueFillBufferData[NUM_ITEMS_clEnqueueFillBuffer] =
{
    {NULL, NULL, NULL, 0, 0, 0, 0, NULL, NULL}
};

const struct clEnqueueCopyBuffer_st clEnqueueCopyBufferData[NUM_ITEMS_clEnqueueCopyBuffer] =
{
    {NULL, NULL, NULL, 0, 0, 0, 0, NULL, NULL}
};

const struct clEnqueueCopyBufferRect_st clEnqueueCopyBufferRectData[NUM_ITEMS_clEnqueueCopyBufferRect] =
{
    {NULL, NULL, NULL, NULL, NULL, NULL, 0, 0, 0, 0, 0, NULL, NULL}
};

const struct clEnqueueMapBuffer_st clEnqueueMapBufferData[NUM_ITEMS_clEnqueueMapBuffer] =
{
    {NULL, NULL, 0, 0, 0, 0, 0, NULL, NULL, NULL}
};

const struct clRetainMemObject_st clRetainMemObjectData[NUM_ITEMS_clRetainMemObject] =
{
    {NULL}
};

const struct clSetMemObjectDestructorCallback_st clSetMemObjectDestructorCallbackData[NUM_ITEMS_clSetMemObjectDestructorCallback] =
{
    {NULL, setmemobjectdestructor_callback, NULL}
};

const struct clEnqueueUnmapMemObject_st clEnqueueUnmapMemObjectData[NUM_ITEMS_clEnqueueUnmapMemObject] =
{
    {NULL, NULL, NULL, 0, NULL, NULL}
};

const struct clGetMemObjectInfo_st clGetMemObjectInfoData[NUM_ITEMS_clGetMemObjectInfo] =
{
    {NULL, 0, 0, NULL, NULL}
};

int test_clEnqueueReadBuffer(const struct clEnqueueReadBuffer_st *data)
{
    test_icd_app_log("clEnqueueReadBuffer(%p, %p, %u, %u, %u, %p, %u, %p, %p)\n",
                     command_queue,
                     buffer,
                     data->blocking_read,
                     data->offset,
                     data->cb,
                     data->ptr,
                     data->num_events_in_wait_list,
                     data->event_wait_list,
                     &event);

    ret_val=clEnqueueReadBuffer(command_queue,
                                buffer,
                                data->blocking_read,
                                data->offset,
                                data->cb,
                                data->ptr,
                                data->num_events_in_wait_list,
                                data->event_wait_list,
                                &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

int test_clEnqueueWriteBuffer(const struct clEnqueueWriteBuffer_st *data)
{
    test_icd_app_log("clEnqueueWriteBuffer(%p, %p, %u, %u, %u, %p, %u, %p, %p)\n",
                     command_queue,
                     buffer,
                     data->blocking_write,
                     data->offset,
                     data->cb,
                     data->ptr,
                     data->num_events_in_wait_list,
                     data->event_wait_list,
                     &event);

    ret_val=clEnqueueWriteBuffer(command_queue,
                                buffer,
                                data->blocking_write,
                                data->offset,
                                data->cb,
                                data->ptr,
                                data->num_events_in_wait_list,
                                data->event_wait_list,
                                &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

int test_clEnqueueReadBufferRect(const struct clEnqueueReadBufferRect_st *data)
{
    test_icd_app_log("clEnqueueReadBufferRect(%p, %p, %u, %p, %p, %p, %u, %u, %u, %u, %p, %u, %p, %p)\n",
                     command_queue,
                     buffer,
                     data->blocking_read,
                     data->buffer_offset,
                     data->host_offset,
                     data->region,
                     data->buffer_row_pitch,
                     data->buffer_slice_pitch,
                     data->host_row_pitch,
                     data->host_slice_pitch,
                     data->ptr,
                     data->num_events_in_wait_list,
                     data->event_wait_list,
                     &event);

    ret_val=clEnqueueReadBufferRect(command_queue,
                                    buffer,
                                    data->blocking_read,
                                    data->buffer_offset,
                                    data->host_offset,
                                    data->region,
                                    data->buffer_row_pitch,
                                    data->buffer_slice_pitch,
                                    data->host_row_pitch,
                                    data->host_slice_pitch,
                                    data->ptr,
                                    data->num_events_in_wait_list,
                                    data->event_wait_list,
                                    &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clEnqueueWriteBufferRect(const struct clEnqueueWriteBufferRect_st *data)
{
    test_icd_app_log("clEnqueueWriteBufferRect(%p, %p, %u, %p, %p, %p, %u, %u, %u, %u, %p, %u, %p, %p)\n",
                     command_queue,
                     buffer,
                     data->blocking_write,
                     data->buffer_offset,
                     data->host_offset,
                     data->region,
                     data->buffer_row_pitch,
                     data->buffer_slice_pitch,
                     data->host_row_pitch,
                     data->host_slice_pitch,
                     data->ptr,
                     data->num_events_in_wait_list,
                     data->event_wait_list,
                     &event);

    ret_val=clEnqueueWriteBufferRect(command_queue,
                                    buffer,
                                    data->blocking_write,
                                    data->buffer_offset,
                                    data->host_offset,
                                    data->region,
                                    data->buffer_row_pitch,
                                    data->buffer_slice_pitch,
                                    data->host_row_pitch,
                                    data->host_slice_pitch,
                                    data->ptr,
                                    data->num_events_in_wait_list,
                                    data->event_wait_list,
                                    &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;
}

int test_clEnqueueFillBuffer(const struct clEnqueueFillBuffer_st *data)
{
    test_icd_app_log("clEnqueueFillBuffer(%p, %p, %p, %u, %u, %u, %u, %p, %p)\n",
                     command_queue,
                     buffer,
                     data->pattern,
                     data->pattern_size,
                     data->offset,
                     data->cb,
                     data->num_events_in_wait_list,
                     data->event_wait_list,
                     &event);

     ret_val=clEnqueueFillBuffer(command_queue,
                                buffer,
                                data->pattern,
                                data->pattern_size,
                                data->offset,
                                data->cb,
                                data->num_events_in_wait_list,
                                data->event_wait_list,
                                &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clEnqueueCopyBuffer(const struct clEnqueueCopyBuffer_st *data)
{
    test_icd_app_log("clEnqueueCopyBuffer(%p, %p, %p, %u, %u, %u, %u, %p, %p)\n",
                     command_queue,
                     data->src_buffer,
                     buffer,
                     data->src_offset,
                     data->dst_offset,
                     data->cb,
                     data->num_events_in_wait_list,
                     data->event_wait_list,
                     &event);

     ret_val=clEnqueueCopyBuffer(command_queue,
                                data->src_buffer,
                                buffer,
                                data->src_offset,
                                data->dst_offset,
                                data->cb,
                                data->num_events_in_wait_list,
                                data->event_wait_list,
                                &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clEnqueueCopyBufferRect(const struct clEnqueueCopyBufferRect_st *data)
{
    test_icd_app_log("clEnqueueCopyBufferRect(%p, %p, %p, %p, %p, %p, %u, %u, %u, %u, %u, %p, %p)\n",
                     command_queue,
                     buffer,
                     buffer,
                     data->src_origin,
                     data->dst_origin,
                     data->region,
                     data->src_row_pitch,
                     data->src_slice_pitch,
                     data->dst_row_pitch,
                     data->dst_slice_pitch,
                     data->num_events_in_wait_list,
                     data->event_wait_list,
                     &event);

    ret_val=clEnqueueCopyBufferRect(command_queue,
                                    buffer,
                                    buffer,
                                    data->src_origin,
                                    data->dst_origin,
                                    data->region,
                                    data->src_row_pitch,
                                    data->src_slice_pitch,
                                    data->dst_row_pitch,
                                    data->dst_slice_pitch,
                                    data->num_events_in_wait_list,
                                    data->event_wait_list,
                                    &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clEnqueueMapBuffer(const struct clEnqueueMapBuffer_st *data)
{
    void * return_value;
    test_icd_app_log("clEnqueueMapBuffer(%p, %p, %u, %x, %u, %u, %u, %p, %p, %p)\n",
                     command_queue,
                     buffer,
                     data->blocking_map,
                     data->map_flags,
                     data->offset,
                     data->cb,
                     data->num_events_in_wait_list,
                     data->event_wait_list,
                     &event,
                     data->errcode_ret);

    return_value=clEnqueueMapBuffer(command_queue,
                                    buffer,
                                    data->blocking_map,
                                    data->map_flags,
                                    data->offset,
                                    data->cb,
                                    data->num_events_in_wait_list,
                                    data->event_wait_list,
                                    &event,
                                    data->errcode_ret);

    test_icd_app_log("Value returned: %p\n", return_value);

    free(return_value);

    return 0;

}

int test_clRetainMemObject(const struct clRetainMemObject_st *data)
{
    test_icd_app_log("clRetainMemObject(%p)\n", buffer);

    ret_val=clRetainMemObject(buffer);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clSetMemObjectDestructorCallback(const struct clSetMemObjectDestructorCallback_st *data)
{
    test_icd_app_log("clSetMemObjectDestructorCallback(%p, %p, %p)\n",
                     buffer,
                     data->pfn_notify,
                     data->user_data);

    ret_val=clSetMemObjectDestructorCallback(buffer,
                                            data->pfn_notify,
                                            data->user_data);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clEnqueueUnmapMemObject(const struct clEnqueueUnmapMemObject_st *data)
{
    test_icd_app_log("clEnqueueUnmapMemObject(%p, %p, %p, %u, %p, %p)\n",
                     command_queue,
                     buffer,
                     data->mapped_ptr,
                     data->num_events_in_wait_list,
                     data->event_wait_list,
                     &event);

    ret_val=clEnqueueUnmapMemObject(command_queue,
                                    buffer,
                                    data->mapped_ptr,
                                    data->num_events_in_wait_list,
                                    data->event_wait_list,
                                    &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}


int test_clGetMemObjectInfo (const struct clGetMemObjectInfo_st *data)
{
    test_icd_app_log("clGetMemObjectInfo(%p, %u, %u, %p, %p)\n",
                      buffer,
                      data->param_name,
                      data->param_value_size,
                      data->param_value,
                      data->param_value_size_ret);

    ret_val=clGetMemObjectInfo(buffer,
                               data->param_name,
                               data->param_value_size,
                               data->param_value,
                               data->param_value_size_ret);

    test_icd_app_log("Value returned: %d\n",ret_val);

    return 0;
}

int test_buffer_object()
{
    int i;
    for (i=0; i<NUM_ITEMS_clEnqueueReadBuffer; i++) {
        test_clEnqueueReadBuffer(&clEnqueueReadBufferData[i]);
    }

    for (i=0; i<NUM_ITEMS_clEnqueueWriteBuffer; i++) {
        test_clEnqueueWriteBuffer(&clEnqueueWriteBufferData[i]);
    }

    for (i=0; i<NUM_ITEMS_clEnqueueReadBufferRect; i++) {
        test_clEnqueueReadBufferRect(&clEnqueueReadBufferRectData[i]);
    }

    for (i=0; i<NUM_ITEMS_clEnqueueWriteBufferRect; i++) {
        test_clEnqueueWriteBufferRect(&clEnqueueWriteBufferRectData[i]);
    }

    for (i=0; i<NUM_ITEMS_clEnqueueFillBuffer; i++) {
        test_clEnqueueFillBuffer(&clEnqueueFillBufferData[i]);
    }

    for (i=0; i<NUM_ITEMS_clEnqueueCopyBuffer; i++) {
        test_clEnqueueCopyBuffer(&clEnqueueCopyBufferData[i]);
    }

    for (i=0; i<NUM_ITEMS_clEnqueueCopyBufferRect; i++) {
        test_clEnqueueCopyBufferRect(&clEnqueueCopyBufferRectData[i]);
    }

    for (i=0; i<NUM_ITEMS_clEnqueueMapBuffer; i++) {
        test_clEnqueueMapBuffer(&clEnqueueMapBufferData[i]);
    }

    for (i=0; i<NUM_ITEMS_clRetainMemObject; i++) {
        test_clRetainMemObject(&clRetainMemObjectData[i]);
    }

    for (i=0; i<NUM_ITEMS_clSetMemObjectDestructorCallback; i++) {
        test_clSetMemObjectDestructorCallback(&clSetMemObjectDestructorCallbackData[i]);
    }

    for (i=0; i<NUM_ITEMS_clEnqueueUnmapMemObject; i++) {
        test_clEnqueueUnmapMemObject(&clEnqueueUnmapMemObjectData[i]);
    }

    for (i=0; i<NUM_ITEMS_clGetMemObjectInfo; i++) {
         test_clGetMemObjectInfo(&clGetMemObjectInfoData[i]);
    }

    return 0;
}

