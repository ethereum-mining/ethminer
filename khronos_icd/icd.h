/*
 * Copyright (c) 2012 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software source and associated documentation files (the "Materials"),
 * to use, copy, modify and compile the Materials to create a binary under the
 * following terms and conditions: 
 *
 * 1. The Materials shall NOT be distributed to any third party;
 *
 * 2. The binary may be distributed without restriction, including without
 * limitation the rights to use, copy, merge, publish, distribute, sublicense,
 * and/or sell copies, and to permit persons to whom the binary is furnished to
 * do so;
 *
 * 3. All modifications to the Materials used to create a binary that is
 * distributed to third parties shall be provided to Khronos with an
 * unrestricted license to use for the purposes of implementing bug fixes and
 * enhancements to the Materials;
 *
 * 4. If the binary is used as part of an OpenCL(TM) implementation, whether
 * binary is distributed together with or separately to that implementation,
 * then recipient must become an OpenCL Adopter and follow the published OpenCL
 * conformance process for that implementation, details at:
 * http://www.khronos.org/conformance/;
 *
 * 5. The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE MATERIALS OR THE USE OR OTHER DEALINGS IN
 * THE MATERIALS.
 * 
 * OpenCL is a trademark of Apple Inc. used under license by Khronos.  
 */

#ifndef _ICD_H_
#define _ICD_H_

#include <CL/cl.h>
#include <CL/cl_ext.h>

/*
 * type definitions
 */

typedef CL_API_ENTRY cl_int (CL_API_CALL *pfn_clIcdGetPlatformIDs)(
    cl_uint num_entries, 
    cl_platform_id *platforms, 
    cl_uint *num_platforms) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *pfn_clGetPlatformInfo)(
    cl_platform_id   platform, 
    cl_platform_info param_name,
    size_t           param_value_size, 
    void *           param_value,
    size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY void *(CL_API_CALL *pfn_clGetExtensionFunctionAddress)(
    const char *function_name)  CL_API_SUFFIX__VERSION_1_0;

typedef struct KHRicdVendorRec KHRicdVendor;
typedef struct KHRicdStateRec  KHRicdState;

/* 
 * KHRicdVendor
 *
 * Data for a single ICD vendor platform.
 */
struct KHRicdVendorRec
{
    // the loaded library object (true type varies on Linux versus Windows)
    void *library;

    // the extension suffix for this platform
    char *suffix;

    // function pointer to the ICD platform IDs extracted from the library
    pfn_clGetExtensionFunctionAddress clGetExtensionFunctionAddress;

    // the platform retrieved from clGetIcdPlatformIDsKHR
    cl_platform_id platform;

    // next vendor in the list vendors
    KHRicdVendor *next;
};


/* 
 * KHRicdState
 *
 * The global state of all vendors
 *
 * TODO: write access to this structure needs to be protected via a mutex
 */

struct KHRicdStateRec 
{
    // has this structure been initialized
    cl_bool initialized;

    // the list of vendors which have been loaded
    KHRicdVendor *vendors;
};

// the global state
extern KHRicdState khrIcdState;

/* 
 * khrIcd interface
 */

// read vendors from system configuration and store the data
// loaded into khrIcdState.  this will call the OS-specific
// function khrIcdEnumerateVendors.  this is called at every
// dispatch function which may be a valid first call into the
// API (e.g, getPlatformIDs, etc).
void khrIcdInitialize(void);

// go through the list of vendors (in /etc/OpenCL.conf or through 
// the registry) and call khrIcdVendorAdd for each vendor encountered
// n.b, this call is OS-specific
void khrIcdOsVendorsEnumerate(void);

// add a vendor's implementation to the list of libraries
void khrIcdVendorAdd(const char *libraryName);

// dynamically load a library.  returns NULL on failure
// n.b, this call is OS-specific
void *khrIcdOsLibraryLoad(const char *libraryName);

// get a function pointer from a loaded library.  returns NULL on failure.
// n.b, this call is OS-specific
void *khrIcdOsLibraryGetFunctionAddress(void *library, const char *functionName);

// unload a library.
// n.b, this call is OS-specific
void khrIcdOsLibraryUnload(void *library);

// parse properties and determine the platform to use from them
void khrIcdContextPropertiesGetPlatform(
    const cl_context_properties *properties, 
    cl_platform_id *outPlatform);

// internal tracing macros
#if 0
    #include <stdio.h>
    #define KHR_ICD_TRACE(...) \
    do \
    { \
        fprintf(stderr, "KHR ICD trace at %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
    } while (0)

    #define KHR_ICD_ASSERT(x) \
    do \
    { \
        if (!(x)) \
        { \
            fprintf(stderr, "KHR ICD assert at %s:%d: %s failed", __FILE__, __LINE__, #x); \
        } \
    } while (0)
#else
    #define KHR_ICD_TRACE(...)
    #define KHR_ICD_ASSERT(x)
#endif

// if handle is NULL then return invalid_handle_error_code
#define KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(handle,invalid_handle_error_code) \
    do \
    { \
        if (!handle) \
        { \
            return invalid_handle_error_code; \
        } \
    } while (0)

// if handle is NULL then set errcode_ret to invalid_handle_error and return NULL 
// (NULL being an invalid handle)
#define KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(handle,invalid_handle_error) \
    do \
    { \
        if (!handle) \
        { \
            if (errcode_ret) \
            { \
                *errcode_ret = invalid_handle_error; \
            } \
            return NULL; \
        } \
    } while (0)


#endif

