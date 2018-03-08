/*
* Wrappers to emulate dlopen() on other systems like Windows
*/

#include "wraphelper.h"

/* assume we can use dlopen itself... */
void *wrap_dlopen(const char *filename) {
	return dlopen(filename, RTLD_NOW);
}
void *wrap_dlsym(void *h, const char *sym) {
	return dlsym(h, sym);
}
int wrap_dlclose(void *h) {
	return dlclose(h);
}
