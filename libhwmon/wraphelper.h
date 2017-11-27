/*
* Wrappers to emulate dlopen() on other systems like Windows
*/
#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
#include <windows.h>
static void *wrap_dlopen(const char *filename) {
	return (void *)LoadLibrary(filename);
}
static void *wrap_dlsym(void *h, const char *sym) {
	return (void *)GetProcAddress((HINSTANCE)h, sym);
}
static int wrap_dlclose(void *h) {
	/* FreeLibrary returns nonzero on success */
	return (!FreeLibrary((HINSTANCE)h));
}
#else
/* assume we can use dlopen itself... */
#include <dlfcn.h>
static void *wrap_dlopen(const char *filename) {
	return dlopen(filename, RTLD_NOW);
}
static void *wrap_dlsym(void *h, const char *sym) {
	return dlsym(h, sym);
}
static int wrap_dlclose(void *h) {
	return dlclose(h);
}
#endif