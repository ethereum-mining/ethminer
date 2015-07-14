add_executable (icd_loader_test
    test_kernel.c
    main.c
    test_platforms.c
    icd_test_match.c
    test_program_objects.c
    test_sampler_objects.c
    test_buffer_object.c
    test_cl_runtime.c
    callbacks.c
    test_create_calls.c
    test_clgl.c
    test_image_objects.c )

target_link_libraries (icd_loader_test OpenCL IcdLog)
