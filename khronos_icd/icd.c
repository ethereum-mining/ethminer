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

#include "icd.h"
#include "icd_dispatch.h"
#include <stdlib.h>
#include <string.h>

KHRicdState khrIcdState = {0};

// entrypoint to initialize the ICD and add all vendors
void khrIcdInitialize(void)
{
    // make sure we don't double-initialize
    // TODO: this should use an atomic exchange to be thread-safe
    if (khrIcdState.initialized)
    {
        return;
    }
    khrIcdState.initialized = CL_TRUE;
    
    // enumerate vendors present on the system
    khrIcdOsVendorsEnumerate();
}

void khrIcdVendorAdd(const char *libraryName)
{
    void *library = NULL;
    cl_int result = CL_SUCCESS;
    pfn_clGetExtensionFunctionAddress p_clGetExtensionFunctionAddress = NULL;
    pfn_clIcdGetPlatformIDs p_clIcdGetPlatformIDs = NULL;
    cl_uint i = 0;
    cl_uint platformCount = 0;
    cl_platform_id *platforms = NULL;

    // require that the library name be valid
    if (!libraryName) 
    {
        goto Done;
    }
    KHR_ICD_TRACE("attempting to add vendor %s...\n", libraryName);

    // load its library and query its function pointers
    library = khrIcdOsLibraryLoad(libraryName);
    if (!library)
    {
        KHR_ICD_TRACE("failed to load library %s\n", libraryName);
        goto Done;
    }

    // get the library's clGetExtensionFunctionAddress pointer
    p_clGetExtensionFunctionAddress = khrIcdOsLibraryGetFunctionAddress(library, "clGetExtensionFunctionAddress");
    if (!p_clGetExtensionFunctionAddress)
    {
        KHR_ICD_TRACE("failed to get function address clGetExtensionFunctionAddress\n");
        goto Done;
    }

    // use that function to get the clIcdGetPlatformIDsKHR function pointer
    p_clIcdGetPlatformIDs = p_clGetExtensionFunctionAddress("clIcdGetPlatformIDsKHR");
    if (!p_clIcdGetPlatformIDs)
    {
        KHR_ICD_TRACE("failed to get extension function address clIcdGetPlatformIDsKHR\n");
        goto Done;
    }

    // query the number of platforms available and allocate space to store them
    result = p_clIcdGetPlatformIDs(0, NULL, &platformCount);
    if (CL_SUCCESS != result)
    {
        KHR_ICD_TRACE("failed clIcdGetPlatformIDs\n");
        goto Done;
    }
    platforms = (cl_platform_id *)malloc(platformCount * sizeof(cl_platform_id) );
    if (!platforms)
    {
        KHR_ICD_TRACE("failed to allocate memory\n");
        goto Done;
    }
    memset(platforms, 0, platformCount * sizeof(cl_platform_id) );
    result = p_clIcdGetPlatformIDs(platformCount, platforms, NULL);
    if (CL_SUCCESS != result)
    {
        KHR_ICD_TRACE("failed clIcdGetPlatformIDs\n");
        goto Done;
    }

    // for each platform, add it
    for (i = 0; i < platformCount; ++i)
    {
        KHRicdVendor* vendor = NULL;
        char *suffix;
        size_t suffixSize;

        // call clGetPlatformInfo on the returned platform to get the suffix
        if (!platforms[i])
        {
            continue;
        }
        result = platforms[i]->dispatch->clGetPlatformInfo(
            platforms[i],
            CL_PLATFORM_ICD_SUFFIX_KHR,
            0,
            NULL,
            &suffixSize);
        if (CL_SUCCESS != result)
        {
            continue;
        }
        suffix = (char *)malloc(suffixSize);
        if (!suffix)
        {
            continue;
        }
        result = platforms[i]->dispatch->clGetPlatformInfo(
            platforms[i],
            CL_PLATFORM_ICD_SUFFIX_KHR,
            suffixSize,
            suffix,
            NULL);            
        if (CL_SUCCESS != result)
        {
            free(suffix);
            continue;
        }

        // allocate a structure for the vendor
        vendor = (KHRicdVendor*)malloc(sizeof(*vendor) );
        if (!vendor) 
        {
            free(suffix);
            KHR_ICD_TRACE("failed to allocate memory\n");
            continue;
        }
        memset(vendor, 0, sizeof(*vendor) );

        // populate vendor data
        vendor->library = khrIcdOsLibraryLoad(libraryName);
        if (!vendor->library) 
        {
            free(suffix);
            free(vendor);
            KHR_ICD_TRACE("failed get platform handle to library\n");
            continue;
        }
        vendor->clGetExtensionFunctionAddress = p_clGetExtensionFunctionAddress;
        vendor->platform = platforms[i];
        vendor->suffix = suffix;

        // add this vendor to the list of vendors at the tail
        {
            KHRicdVendor **prevNextPointer = NULL;
            for (prevNextPointer = &khrIcdState.vendors; *prevNextPointer; prevNextPointer = &( (*prevNextPointer)->next) );
            *prevNextPointer = vendor;
        }

        KHR_ICD_TRACE("successfully added vendor %s with suffix %s\n", libraryName, suffix);

    }

Done:

    if (library)
    {
        khrIcdOsLibraryUnload(library);
    }
}

void khrIcdContextPropertiesGetPlatform(const cl_context_properties *properties, cl_platform_id *outPlatform)
{
    const cl_context_properties *property = (cl_context_properties *)NULL;
    *outPlatform = NULL;
    for (property = properties; property && property[0]; property += 2)
    {
        if ((cl_context_properties)CL_CONTEXT_PLATFORM == property[0])
        {
            *outPlatform = (cl_platform_id)property[1];
        }    
    }
}

