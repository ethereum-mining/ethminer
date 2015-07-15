#include <CL/cl.h>
#include "param_struct.h"
#include <platform/icd_test_log.h>

extern cl_mem image;
extern cl_context  context;
extern cl_command_queue command_queue;
extern cl_event event;
extern cl_mem buffer;

int ret_val;

const struct clGetSupportedImageFormats_st clGetSupportedImageFormatsData[NUM_ITEMS_clGetSupportedImageFormats] =
{
    { NULL, 0x0, 0, 0, NULL, NULL }
};

const struct clEnqueueCopyImageToBuffer_st clEnqueueCopyImageToBufferData[NUM_ITEMS_clEnqueueCopyImageToBuffer] =
{
    { NULL, NULL, NULL, NULL, NULL, 0, 0, NULL, NULL }
};

const struct clEnqueueCopyBufferToImage_st clEnqueueCopyBufferToImageData[NUM_ITEMS_clEnqueueCopyBufferToImage] =
{
    { NULL, NULL, NULL, 0, NULL, NULL, 0, NULL, NULL }
};

const struct clEnqueueMapImage_st clEnqueueMapImageData[NUM_ITEMS_clEnqueueMapImage] =
{
    { NULL, NULL, 0, 0x0, NULL, NULL, NULL, NULL,0, NULL, NULL}
};

const struct clEnqueueReadImage_st clEnqueueReadImageData[NUM_ITEMS_clEnqueueReadImage] =
{
    { NULL, NULL, 0, NULL, NULL, 0, 0, NULL, 0, NULL, NULL }
};

const struct clEnqueueWriteImage_st clEnqueueWriteImageData[NUM_ITEMS_clEnqueueWriteImage] =
{
    { NULL, NULL, 0, NULL, NULL, 0, 0, NULL, 0, NULL, NULL }
};

const struct clEnqueueFillImage_st clEnqueueFillImageData[NUM_ITEMS_clEnqueueFillImage] =
{
    { NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL }
};

const struct clEnqueueCopyImage_st clEnqueueCopyImageData[NUM_ITEMS_clEnqueueCopyImage] =
{
    { NULL, NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL }
};

const struct clGetImageInfo_st clGetImageInfoData[NUM_ITEMS_clGetImageInfo] =
{
    { NULL, 0, 0, NULL, NULL}
};

int test_clGetSupportedImageFormats(const struct clGetSupportedImageFormats_st *data)
{
    test_icd_app_log("clGetSupportedImageFormats(%p, %x, %u, %u, %p, %p)\n",
                     context,
                     data->flags,
                     data->image_type,
                     data->num_entries,
                     data->image_formats,
                     data->num_image_formats);
    
    ret_val = clGetSupportedImageFormats(context,
                                    data->flags,
                                    data->image_type,
                                    data->num_entries,
                                    data->image_formats,
                                    data->num_image_formats);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clEnqueueCopyImageToBuffer(const struct clEnqueueCopyImageToBuffer_st *data)
{
    test_icd_app_log("clEnqueueCopyImageToBuffer(%p, %p, %p, %p, %p, %u, %u, %p, %p)\n",
                     command_queue,
                     image,
                     buffer,
                     data->src_origin,
                     data->region,
                     data->dst_offset,
                     data->num_events_in_wait_list,
                     data->event_wait_list,
                     &event);

    ret_val = clEnqueueCopyImageToBuffer(command_queue,
                                    image,
                                    buffer,
                                    data->src_origin,
                                    data->region,
                                    data->dst_offset,
                                    data->num_events_in_wait_list,
                                    data->event_wait_list,
                                    &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clEnqueueCopyBufferToImage(const struct clEnqueueCopyBufferToImage_st *data)
{
    test_icd_app_log("clEnqueueCopyBufferToImage(%p, %p, %p, %u, %p, %p, %u, %p, %p)\n",
                     command_queue,
                     buffer,
                     image,
                     data->src_offset,
                     data->dst_origin,
                     data->region,
                     data->num_events_in_wait_list,
                     data->event_wait_list,
                     &event);

    ret_val = clEnqueueCopyBufferToImage(command_queue,
                                    buffer,
                                    image,
                                    data->src_offset,
                                    data->dst_origin,
                                    data->region,
                                    data->num_events_in_wait_list,
                                    data->event_wait_list,
                                    &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clEnqueueMapImage(const struct clEnqueueMapImage_st *data)
{
    void *return_value;
    test_icd_app_log("clEnqueueMapImage(%p, %p, %u, %x, %p, %p, %p, %p, %u, %p, %p, %p)\n",
                     command_queue,
                     image,
                     data->blocking_map,
                     data->map_flags,
                     data->origin,
                     data->region,
                     data->image_row_pitch,
                     data->image_slice_pitch,
                     data->num_events_in_wait_list,
                     data->event_wait_list,
                     &event,
                     data->errcode_ret);

    return_value = clEnqueueMapImage(command_queue,
                                image,
                                data->blocking_map,
                                data->map_flags,
                                data->origin,
                                data->region,
                                data->image_row_pitch,
                                data->image_slice_pitch,
                                data->num_events_in_wait_list,
                                data->event_wait_list,
                                &event,
                                data->errcode_ret);

    test_icd_app_log("Value returned: %p\n", return_value);

    free(return_value);

    return 0;

}

int test_clEnqueueReadImage(const struct clEnqueueReadImage_st *data)
{
    test_icd_app_log("clEnqueueReadImage(%p, %p, %u, %p, %p, %u, %u, %p, %u, %p, %p)\n",
                     command_queue,
                     image,
                     data->blocking_read,
                     data->origin,            
                     data->region,
                     data->row_pitch,
                     data->slice_pitch,
                     data->ptr,
                     data->num_events_in_wait_list,
                     data->event_wait_list,
                     &event);

    ret_val = clEnqueueReadImage(command_queue,
                            image,
                            data->blocking_read,
                            data->origin,
                            data->region, 
                            data->row_pitch,  
                            data->slice_pitch, 
                            data->ptr,       
                            data->num_events_in_wait_list,  
                            data->event_wait_list,
                            &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clEnqueueWriteImage(const struct clEnqueueWriteImage_st *data)
{
    test_icd_app_log("clEnqueueWriteImage(%p, %p, %u, %p, %p, %u, %u, %p, %u, %p, %p)\n",
                     command_queue,
                     image,
                     data->blocking_write,
                     data->origin,
                     data->region,
                     data->input_row_pitch,
                     data->input_slice_pitch,
                     data->ptr,
                     data->num_events_in_wait_list,
                     data->event_wait_list,
                     &event);

    ret_val = clEnqueueWriteImage(command_queue,
                                image,
                                data->blocking_write,
                                data->origin,
                                data->region,
                                data->input_row_pitch,
                                data->input_slice_pitch,
                                data->ptr,
                                data->num_events_in_wait_list,
                                data->event_wait_list,
                                &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_clEnqueueFillImage(const struct clEnqueueFillImage_st *data)
{
    test_icd_app_log("clEnqueueFillImage(%p, %p, %p, %p, %p, %u, %p, %p)\n",
                     command_queue,
                     image,
                     data->fill_color,
                     data->origin,
                     data->region,
                     data->num_events_in_wait_list,
                     data->event_wait_list, 
                     &event);

    ret_val = clEnqueueFillImage(command_queue,
                            image,    
                            data->fill_color,
                            data->origin,
                            data->region,
                            data->num_events_in_wait_list,
                            data->event_wait_list, 
                            &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}
int test_clEnqueueCopyImage(const struct clEnqueueCopyImage_st *data)
{
    test_icd_app_log("clEnqueueCopyImage(%p, %p, %p, %p, %p, %p, %u, %p, %p)\n",
                     command_queue,
                     image,
                     image,
                     data->src_origin,
                     data->dst_origin,
                     data->region,
                     data->num_events_in_wait_list,
                     data->event_wait_list, 
                     &event);

    ret_val = clEnqueueCopyImage(command_queue,
                            image,
                            image,
                            data->src_origin,
                            data->dst_origin,
                            data->region,
                            data->num_events_in_wait_list,
                            data->event_wait_list, 
                            &event);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}


int test_clGetImageInfo(const struct clGetImageInfo_st *data)
{
    test_icd_app_log("clGetImageInfo(%p, %u, %u, %p, %p)\n",
                     image,
                     data->param_name,
                     data->param_value_size,
                     data->param_value,
                     data->param_value_size_ret);

    ret_val = clGetImageInfo(image,
                        data->param_name,
                        data->param_value_size,
                        data->param_value,
                        data->param_value_size_ret);

    test_icd_app_log("Value returned: %d\n", ret_val);

    return 0;

}

int test_image_objects()
{
    int i;

    for (i = 0; i<NUM_ITEMS_clGetSupportedImageFormats; i++) {
        test_clGetSupportedImageFormats(&clGetSupportedImageFormatsData[i]);
    }

    for (i = 0; i<NUM_ITEMS_clEnqueueCopyImageToBuffer; i++) {
        test_clEnqueueCopyImageToBuffer(&clEnqueueCopyImageToBufferData[i]);
    }

    for (i = 0; i<NUM_ITEMS_clEnqueueCopyBufferToImage; i++) {
        test_clEnqueueCopyBufferToImage(&clEnqueueCopyBufferToImageData[i]);
    }

    for (i = 0; i<NUM_ITEMS_clEnqueueMapImage; i++) {
        test_clEnqueueMapImage(&clEnqueueMapImageData[i]);
    }

    for (i = 0; i<NUM_ITEMS_clEnqueueReadImage; i++) {
        test_clEnqueueReadImage(&clEnqueueReadImageData[i]);
    }

    for (i = 0; i<NUM_ITEMS_clEnqueueWriteImage; i++) {
        test_clEnqueueWriteImage(&clEnqueueWriteImageData[i]);
    }

    for (i = 0; i<NUM_ITEMS_clEnqueueFillImage; i++) {
        test_clEnqueueFillImage(&clEnqueueFillImageData[i]);
    }

    for (i = 0; i<NUM_ITEMS_clEnqueueCopyImage; i++) {
        test_clEnqueueCopyImage(&clEnqueueCopyImageData[i]);
    }

    for (i = 0; i<NUM_ITEMS_clGetImageInfo; i++) {
        test_clGetImageInfo(&clGetImageInfoData[i]);
    }

    return 0;

}
