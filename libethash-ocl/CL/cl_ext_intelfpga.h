/******   Supported 1.1 Features    *******/
//#define CL_API_SUFFIX__VERSION_1_1
//#define CL_EXT_SUFFIX__VERSION_1_1
//
//typedef cl_uint             cl_buffer_create_type;
//
//typedef struct _cl_buffer_region {
//    size_t                  origin;
//    size_t                  size;
//} cl_buffer_region;
//
//#define CL_MISALIGNED_SUB_BUFFER_OFFSET             -13
//#define CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST -14
//
//#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF       0x1034
//#define CL_DEVICE_HOST_UNIFIED_MEMORY               0x1035
//#define CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR          0x1036
//#define CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT         0x1037
//#define CL_DEVICE_NATIVE_VECTOR_WIDTH_INT           0x1038
//#define CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG          0x1039
//#define CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT         0x103A
//#define CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE        0x103B
//#define CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF          0x103C
//#define CL_DEVICE_OPENCL_C_VERSION                  0x103D
//
///* cl_context_info  */
//#define CL_CONTEXT_NUM_DEVICES                      0x1083
//
///* cl_mem_info */
//#define CL_MEM_ASSOCIATED_MEMOBJECT                 0x1107
//#define CL_MEM_OFFSET                               0x1108
//
///* cl_addressing_mode */
//#define CL_ADDRESS_MIRRORED_REPEAT                  0x1134
//
///* cl_command_type */
//#define CL_COMMAND_READ_BUFFER_RECT                 0x1201
//#define CL_COMMAND_WRITE_BUFFER_RECT                0x1202
//#define CL_COMMAND_COPY_BUFFER_RECT                 0x1203
//#define CL_COMMAND_USER                             0x1204
//
///* cl_buffer_create_type  */
//#define CL_BUFFER_CREATE_TYPE_REGION                0x1220
//
///* Support for sub buffers */
//extern CL_API_ENTRY cl_mem CL_API_CALL
//clCreateSubBuffer(cl_mem                   /* buffer */,
//                  cl_mem_flags             /* flags */,
//                  cl_buffer_create_type    /* buffer_create_type */,
//                  const void *             /* buffer_create_info */,
//                  cl_int *                 /* errcode_ret */) CL_API_SUFFIX__VERSION_1_1;
//
///* Support for user events */
//extern CL_API_ENTRY cl_event CL_API_CALL
//clCreateUserEvent(cl_context    /* context */,
//                  cl_int *      /* errcode_ret */) CL_API_SUFFIX__VERSION_1_1;               
//
//extern CL_API_ENTRY cl_int CL_API_CALL
//clSetUserEventStatus(cl_event   /* event */,
//                     cl_int     /* execution_status */) CL_API_SUFFIX__VERSION_1_1;
//                     
//extern CL_API_ENTRY cl_int CL_API_CALL
//clEnqueueReadBufferRect(cl_command_queue    /* command_queue */,
//                        cl_mem              /* buffer */,
//                        cl_bool             /* blocking_read */,
//                        const size_t *      /* buffer_offset */,
//                        const size_t *      /* host_offset */, 
//                        const size_t *      /* region */,
//                        size_t              /* buffer_row_pitch */,
//                        size_t              /* buffer_slice_pitch */,
//                        size_t              /* host_row_pitch */,
//                        size_t              /* host_slice_pitch */,                        
//                        void *              /* ptr */,
//                        cl_uint             /* num_events_in_wait_list */,
//                        const cl_event *    /* event_wait_list */,
//                        cl_event *          /* event */) CL_API_SUFFIX__VERSION_1_1;
//                            
// extern CL_API_ENTRY cl_int CL_API_CALL
//clEnqueueWriteBufferRect(cl_command_queue    /* command_queue */,
//                         cl_mem              /* buffer */,
//                         cl_bool             /* blocking_write */,
//                         const size_t *      /* buffer_offset */,
//                         const size_t *      /* host_offset */, 
//                         const size_t *      /* region */,
//                         size_t              /* buffer_row_pitch */,
//                         size_t              /* buffer_slice_pitch */,
//                         size_t              /* host_row_pitch */,
//                         size_t              /* host_slice_pitch */,                        
//                         const void *        /* ptr */,
//                         cl_uint             /* num_events_in_wait_list */,
//                         const cl_event *    /* event_wait_list */,
//                         cl_event *          /* event */) CL_API_SUFFIX__VERSION_1_1;
//                            
// extern CL_API_ENTRY cl_int CL_API_CALL
//clEnqueueCopyBufferRect(cl_command_queue    /* command_queue */, 
//                        cl_mem              /* src_buffer */,
//                        cl_mem              /* dst_buffer */, 
//                        const size_t *      /* src_origin */,
//                        const size_t *      /* dst_origin */,
//                        const size_t *      /* region */, 
//                        size_t              /* src_row_pitch */,
//                        size_t              /* src_slice_pitch */,
//                        size_t              /* dst_row_pitch */,
//                        size_t              /* dst_slice_pitch */,
//                        cl_uint             /* num_events_in_wait_list */,
//                        const cl_event *    /* event_wait_list */,
//                        cl_event *          /* event */) CL_API_SUFFIX__VERSION_1_1;
//                            

/* cl_kernel_work_group_info */
#define CL_KERNEL_GLOBAL_WORK_SIZE                  0x11B5

extern
CL_API_ENTRY cl_int clSetEventCallback (cl_event event,
    cl_int command_exec_callback_type,
    void (CL_CALLBACK *pfn_event_notify)(cl_event event,
    cl_int event_command_exec_status, void *user_data),
    void *user_data);

extern
CL_API_ENTRY cl_int clSetMemObjectDestructorCallback (cl_mem memobj,
               void (CL_CALLBACK *pfn_notify)(cl_mem memobj, void *user_data),
               void *user_data);

/******   Supported 1.2 Features    *******/
#define CL_API_SUFFIX__VERSION_1_2
#define CL_EXT_SUFFIX__VERSION_1_2

#define CL_DEVICE_PRINTF_BUFFER_SIZE                0x1049

#define CL_COMMAND_MIGRATE_MEM_OBJECTS              0x1206
typedef cl_bitfield         cl_mem_migration_flags;

/* cl_program_info */
#define CL_PROGRAM_NUM_KERNELS                      0x1167
#define CL_PROGRAM_KERNEL_NAMES                     0x1168

/* cl_program_build_info */
#define CL_PROGRAM_BINARY_TYPE 0x1184
#define CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE 0x1185

typedef cl_uint cl_program_binary_type;   
#define CL_PROGRAM_BINARY_TYPE_NONE                0x0
#define CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT     0x1
#define CL_PROGRAM_BINARY_TYPE_LIBRARY             0x2
#define CL_PROGRAM_BINARY_TYPE_EXECUTABLE          0x3

/* cl_get_kernel_arg_info */
typedef cl_uint cl_kernel_arg_info;
typedef cl_uint cl_kernel_arg_address_qualifier;
typedef cl_uint cl_kernel_arg_access_qualifier;
typedef cl_bitfield cl_kernel_arg_type_qualifier;

/* cl_kernel_arg_info */
#define CL_KERNEL_ARG_ADDRESS_QUALIFIER             0x1196
#define CL_KERNEL_ARG_ACCESS_QUALIFIER              0x1197
#define CL_KERNEL_ARG_TYPE_NAME                     0x1198
#define CL_KERNEL_ARG_TYPE_QUALIFIER                0x1199
#define CL_KERNEL_ARG_NAME                          0x119A
#define CL_KERNEL_ARG_INFO_NOT_AVAILABLE            -19

/* cl_kernel_arg_address_qualifier */
#define CL_KERNEL_ARG_ADDRESS_GLOBAL                0x119B
#define CL_KERNEL_ARG_ADDRESS_LOCAL                 0x119C
#define CL_KERNEL_ARG_ADDRESS_CONSTANT              0x119D
#define CL_KERNEL_ARG_ADDRESS_PRIVATE               0x119E

/* cl_kernel_arg_access_qualifier */
#define CL_KERNEL_ARG_ACCESS_READ_ONLY              0x11A0
#define CL_KERNEL_ARG_ACCESS_WRITE_ONLY             0x11A1
#define CL_KERNEL_ARG_ACCESS_READ_WRITE             0x11A2
#define CL_KERNEL_ARG_ACCESS_NONE                   0x11A3
    
//The first 4 cases can be used in combination as flags.    
/* cl_kernel_arg_type_qualifer */
#define CL_KERNEL_ARG_TYPE_NONE                     0
#define CL_KERNEL_ARG_TYPE_CONST                    (1 << 0)
#define CL_KERNEL_ARG_TYPE_RESTRICT                 (1 << 1)
#define CL_KERNEL_ARG_TYPE_VOLATILE                 (1 << 2)
#define CL_KERNEL_ARG_TYPE_PIPE                     (1 << 3)


extern
CL_API_ENTRY cl_int clGetKernelArgInfo (cl_kernel kernel,
    cl_uint arg_indx,
    cl_kernel_arg_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret);

extern
CL_API_ENTRY cl_int clEnqueueFillBuffer (
               cl_command_queue command_queue,
               cl_mem buffer,
               const void *pattern,
               size_t pattern_size,
               size_t offset,
               size_t size,
               cl_uint num_events_in_wait_list,
               const cl_event *event_wait_list,
               cl_event *event);

extern
CL_API_ENTRY CL_API_ENTRY cl_int CL_API_CALL clEnqueueFillImage (cl_command_queue command_queue,
               cl_mem image,
               const void *fill_color,
               const size_t *origin,
               const size_t *region,
               cl_uint num_events_in_wait_list,
               const cl_event *event_wait_list,
               cl_event *event);

/* cl_mem_migration_flags - bitfield */
#define CL_MIGRATE_MEM_OBJECT_HOST                  (1 << 0)
#define CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED     (1 << 1)

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMigrateMemObjects(cl_command_queue       /* command_queue */,
                           cl_uint                /* num_mem_objects */,
                           const cl_mem *         /* mem_objects */,
                           cl_mem_migration_flags /* flags */,
                           cl_uint                /* num_events_in_wait_list */,
                           const cl_event *       /* event_wait_list */,
                           cl_event *             /* event */) CL_API_SUFFIX__VERSION_1_2;

/* Samplers */
typedef cl_bitfield         cl_sampler_properties;

extern CL_API_ENTRY cl_sampler CL_API_CALL
clCreateSamplerWithProperties(cl_context                     /* context */,
                                      const cl_sampler_properties *  /* normalized_coords */,
                                                                    cl_int *                       /* errcode_ret */);

#define CL_DEVICE_IMAGE_MAX_ARRAY_SIZE                  0x1041

/******   Supported 2.0 Features    *******/
#define CL_API_SUFFIX__VERSION_2_0
#define CL_EXT_SUFFIX__VERSION_2_0

/******************************************
 * SVM extentions
 */
// OpenCL 2.0 cl.h definitions
typedef cl_bitfield         cl_device_svm_capabilities;
typedef cl_bitfield         cl_svm_mem_flags;
typedef cl_bitfield         cl_queue_properties;

#define CL_MEM_SVM_FINE_GRAIN_BUFFER                (1 << 10)   /* used by cl_svm_mem_flags only */
#define CL_MEM_SVM_ATOMICS                          (1 << 11)   /* used by cl_svm_mem_flags only */

#define CL_DEVICE_SVM_CAPABILITIES                      0x1053

#define CL_MEM_USES_SVM_POINTER                     0x1109

/* cl_device_svm_capabilities */
#define CL_DEVICE_SVM_COARSE_GRAIN_BUFFER           (1 << 0)
#define CL_DEVICE_SVM_FINE_GRAIN_BUFFER             (1 << 1)
#define CL_DEVICE_SVM_FINE_GRAIN_SYSTEM             (1 << 2)
#define CL_DEVICE_SVM_ATOMICS                       (1 << 3)

#define CL_COMMAND_SVM_FREE                         0x1209
#define CL_COMMAND_SVM_MEMCPY                       0x120A
#define CL_COMMAND_SVM_MEMFILL                      0x120B
#define CL_COMMAND_SVM_MAP                          0x120C
#define CL_COMMAND_SVM_UNMAP                        0x120D

/* cl_command_queue_info */
#define CL_QUEUE_SIZE                               0x1094

/* cl_command_queue_properties - bitfield */
#define CL_QUEUE_ON_DEVICE                          (1 << 2)
#define CL_QUEUE_ON_DEVICE_DEFAULT                  (1 << 3)
#define CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE        0x104F
#define CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE              0x1050

/* clGetDeviceInfo additional param names */
#define CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS              0x104C
#define CL_DEVICE_IMAGE_MAX_BUFFER_SIZE                  0x1040
#define CL_DEVICE_IMAGE_PITCH_ALIGNMENT                  0x104A
#define CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT           0x104B
#define CL_DEVICE_LINKER_AVAILABLE                       0x103E
#define CL_DEVICE_QUEUE_ON_HOST_PROPERTIES               0x102A
#define CL_DEVICE_BUILT_IN_KERNELS                       0x103F
#define CL_DEVICE_PREFERRED_INTEROP_USER_SYNC            0x1048
#define CL_DEVICE_PARENT_DEVICE                          0x1042
#define CL_DEVICE_PARTITION_MAX_SUB_DEVICES              0x1043
#define CL_DEVICE_PARTITION_PROPERTIES                   0x1044
#define CL_DEVICE_PARTITION_AFFINITY_DOMAIN              0x1045
#define CL_DEVICE_PARTITION_TYPE                         0x1046
#define CL_DEVICE_REFERENCE_COUNT                        0x1047
#define CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT    0x1058
#define CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT      0x1059
#define CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT       0x105A


//////////////////////////////
// OpenCL API

/* SVM Allocation APIs */
extern CL_API_ENTRY void * CL_API_CALL
clSVMAlloc(cl_context       /* context */,
           cl_svm_mem_flags /* flags */,
           size_t           /* size */,
           cl_uint          /* alignment */);

extern CL_API_ENTRY void CL_API_CALL
clSVMFree(cl_context        /* context */,
          void *            /* svm_pointer */);
    
extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMFree(cl_command_queue  /* command_queue */,
                 cl_uint           /* num_svm_pointers */,
                 void *[]          /* svm_pointers[] */,
                 void (CL_CALLBACK * /*pfn_free_func*/)(cl_command_queue /* queue */,
                                                        cl_uint          /* num_svm_pointers */,
                                                        void *[]         /* svm_pointers[] */,
                                                        void *           /* user_data */),
                 void *            /* user_data */,
                 cl_uint           /* num_events_in_wait_list */,
                 const cl_event *  /* event_wait_list */,
                 cl_event *        /* event */);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMemcpy(cl_command_queue  /* command_queue */,
                   cl_bool           /* blocking_copy */,
                   void *            /* dst_ptr */,
                   const void *      /* src_ptr */,
                   size_t            /* size */,
                   cl_uint           /* num_events_in_wait_list */,
                   const cl_event *  /* event_wait_list */,
                   cl_event *        /* event */);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMemFill(cl_command_queue  /* command_queue */,
                    void *            /* svm_ptr */,
                    const void *      /* pattern */,
                    size_t            /* pattern_size */,
                    size_t            /* size */,
                    cl_uint           /* num_events_in_wait_list */,
                    const cl_event *  /* event_wait_list */,
                    cl_event *        /* event */);
    
extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMap(cl_command_queue  /* command_queue */,
                cl_bool           /* blocking_map */,
                cl_map_flags      /* flags */,
                void *            /* svm_ptr */,
                size_t            /* size */,
                cl_uint           /* num_events_in_wait_list */,
                const cl_event *  /* event_wait_list */,
                cl_event *        /* event */);
    
extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMUnmap(cl_command_queue  /* command_queue */,
                  void *            /* svm_ptr */,
                  cl_uint           /* num_events_in_wait_list */,
                  const cl_event *  /* event_wait_list */,
                  cl_event *        /* event */);

extern CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArgSVMPointer (cl_kernel      /* kernel */,
                                 cl_uint        /* arg_index */,
                                 const void *   /* arg_value */);
/* Internal SVM Allocation APIs */
extern CL_API_ENTRY void * CL_API_CALL
clSVMAllocIntelFPGA(cl_context       /* context */,
           cl_svm_mem_flags /* flags */,
           size_t           /* size */,
           cl_uint          /* alignment */);

extern CL_API_ENTRY void CL_API_CALL
clSVMFreeIntelFPGA(cl_context        /* context */,
          void *            /* svm_pointer */);
    
extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMFreeIntelFPGA(cl_command_queue  /* command_queue */,
                 cl_uint           /* num_svm_pointers */,
                 void *[]          /* svm_pointers[] */,
                 void (CL_CALLBACK * /*pfn_free_func*/)(cl_command_queue /* queue */,
                                                        cl_uint          /* num_svm_pointers */,
                                                        void *[]         /* svm_pointers[] */,
                                                        void *           /* user_data */),
                 void *            /* user_data */,
                 cl_uint           /* num_events_in_wait_list */,
                 const cl_event *  /* event_wait_list */,
                 cl_event *        /* event */);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMemcpyIntelFPGA(cl_command_queue  /* command_queue */,
                   cl_bool           /* blocking_copy */,
                   void *            /* dst_ptr */,
                   const void *      /* src_ptr */,
                   size_t            /* size */,
                   cl_uint           /* num_events_in_wait_list */,
                   const cl_event *  /* event_wait_list */,
                   cl_event *        /* event */);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMemFillIntelFPGA(cl_command_queue  /* command_queue */,
                    void *            /* svm_ptr */,
                    const void *      /* pattern */,
                    size_t            /* pattern_size */,
                    size_t            /* size */,
                    cl_uint           /* num_events_in_wait_list */,
                    const cl_event *  /* event_wait_list */,
                    cl_event *        /* event */);
    
extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMapIntelFPGA(cl_command_queue  /* command_queue */,
                cl_bool           /* blocking_map */,
                cl_map_flags      /* flags */,
                void *            /* svm_ptr */,
                size_t            /* size */,
                cl_uint           /* num_events_in_wait_list */,
                const cl_event *  /* event_wait_list */,
                cl_event *        /* event */);
    
extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMUnmapIntelFPGA(cl_command_queue  /* command_queue */,
                  void *            /* svm_ptr */,
                  cl_uint           /* num_events_in_wait_list */,
                  const cl_event *  /* event_wait_list */,
                  cl_event *        /* event */);

extern CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArgSVMPointerIntelFPGA (cl_kernel      /* kernel */,
                                 cl_uint        /* arg_index */,
                                 const void *   /* arg_value */);

extern CL_API_ENTRY void CL_API_CALL
clSetBoardLibraryIntelFPGA  (char* /* library_name */);

extern CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueueWithProperties(cl_context               /* context */,
                                   cl_device_id             /* device */,
                                   const cl_queue_properties *    /* properties */,
                                   cl_int *                 /* errcode_ret */);

/* Image Support */

// Ignore warning about nameless union
#pragma warning( push )
#pragma warning( disable:4201 )
typedef struct _cl_image_desc {
    cl_mem_object_type      image_type;
    size_t                  image_width;
    size_t                  image_height;
    size_t                  image_depth;
    size_t                  image_array_size;
    size_t                  image_row_pitch;
    size_t                  image_slice_pitch;
    cl_uint                 num_mip_levels;
    cl_uint                 num_samples;
    union {
      cl_mem                  buffer;
      cl_mem                  mem_object;
    };
} cl_image_desc;
#pragma warning( pop )

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateImage (cl_context /* context */,
               cl_mem_flags /* flags */,
               const cl_image_format * /* image_format */,
               const cl_image_desc * /* image_desc */,
               void * /* host_ptr */,
               cl_int * /* errcode_ret */);

#define CL_MEM_OBJECT_IMAGE2D_ARRAY                 0x10F3
#define CL_MEM_OBJECT_IMAGE1D                       0x10F4
#define CL_MEM_OBJECT_IMAGE1D_ARRAY                 0x10F5
#define CL_MEM_OBJECT_IMAGE1D_BUFFER                0x10F6

#define CL_INVALID_IMAGE_DESCRIPTOR                 -65

/* Pipe Support */

typedef intptr_t            cl_pipe_properties;
typedef cl_uint             cl_pipe_info;

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreatePipe(cl_context                 /* context */,
             cl_mem_flags               /* flags */,
             cl_uint                    /* pipe_packet_size */,
             cl_uint                    /* pipe_max_packets */,
             const cl_pipe_properties * /* properties */,
             cl_int *                   /* errcode_ret */);

extern CL_API_ENTRY cl_int CL_API_CALL
clGetPipeInfo(cl_mem           /* pipe */,
              cl_pipe_info     /* param_name */,
              size_t           /* param_value_size */,
              void *           /* param_value */,
              size_t *         /* param_value_size_ret */);

#define CL_MEM_OBJECT_PIPE                          0x10F7

#define CL_DEVICE_MAX_PIPE_ARGS                         0x1055
#define CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS          0x1056
#define CL_DEVICE_PIPE_MAX_PACKET_SIZE                  0x1057

/* cl_pipe_info */
#define CL_PIPE_PACKET_SIZE                         0x1120
#define CL_PIPE_MAX_PACKETS                         0x1121

#define CL_INVALID_PIPE_SIZE                        -69

/******   Supported 2.1 Features    *******/

