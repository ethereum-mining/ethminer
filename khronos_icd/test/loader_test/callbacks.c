#include <CL/cl.h>
#include <stdio.h>
#include <platform/icd_test_log.h>

void CL_CALLBACK createcontext_callback(const char* _a, const void* _b, size_t _c, void* _d)
{
    test_icd_app_log("createcontext_callback(%p, %p, %u, %p)\n", 
                    _a, 
                    _b, 
                    _c, 
                    _d);
}

void CL_CALLBACK setmemobjectdestructor_callback(cl_mem _a, void* _b)
{
    test_icd_app_log("setmemobjectdestructor_callback(%p, %p)\n", 
                    _a, 
                    _b);
}

void CL_CALLBACK program_callback(cl_program _a, void* _b)
{
    test_icd_app_log("program_callback(%p, %p)\n", 
                    _a, 
                    _b);
}

void CL_CALLBACK setevent_callback(cl_event _a, cl_int _b, void* _c)
{
    test_icd_app_log("setevent_callback(%p, %d, %p)\n", 
                    _a, 
                    _b, 
                    _c);
}

void CL_CALLBACK setprintf_callback(cl_context _a, cl_uint _b, char* _c, void* _d )
{
    test_icd_app_log("setprintf_callback(%p, %u, %p, %p)\n", 
                    _a, 
                    _b, 
                    _c, 
                    _d);
}
