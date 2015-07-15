#include <string.h>

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include "CL/cl.h"
#include "CL/cl_ext.h"

struct driverStubextFunc_st 
{
    const char *name;
    void *func;
};

#define EXT_FUNC(name) { #name, (void*)(name) }

static struct driverStubextFunc_st clExtensions[] = 
{
    EXT_FUNC(clIcdGetPlatformIDsKHR),
};

static const int clExtensionCount = sizeof(clExtensions) / sizeof(clExtensions[0]);

CL_API_ENTRY void * CL_API_CALL
clGetExtensionFunctionAddress(const char *name)
{
    int ii;
    
    for (ii = 0; ii < clExtensionCount; ii++) {
        if (!strcmp(name, clExtensions[ii].name)) {
            return clExtensions[ii].func;
        }
    }

    return NULL;
}

