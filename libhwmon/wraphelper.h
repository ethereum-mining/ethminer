/*
* Wrappers to emulate dlopen() on other systems like Windows
*/

#pragma once

#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
#include <windows.h>
void *wrap_dlopen(const char *filename);
void *wrap_dlsym(void *h, const char *sym);
int wrap_dlclose(void *h);
#else
/* assume we can use dlopen itself... */
#include <dlfcn.h>
void *wrap_dlopen(const char *filename);
void *wrap_dlsym(void *h, const char *sym);
int wrap_dlclose(void *h);
#endif
