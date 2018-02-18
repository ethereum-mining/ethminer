/*
* Wrappers to emulate dlopen() on other systems like Windows
*/

#pragma once

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
//Converts from 8 fractional bit int to double (ADL method result format)
static double adlf2double(int adlf){
	double res = 0.0;
	int fraction = 	adlf & 0x00FF; //last 8 bits seem to be decimal part
	int number = 	(adlf & 0xFF00) >> 8; //first 8 bits are the integer part
	double dfraction = 0; //decimal part
	int i = 1;
	while(fraction > 0){
		double digit = number % 10;
		if(fraction >= 100){//256 max value, if 3 digits, divide by 1000
			dfraction += (digit / (double)1000.0);
		}
		else if(fraction >=10){//if 2 digits...
			dfraction += (digit / (double)100.0);
		}
		else {//1 digit
			dfraction += (digit / (double)10.0);
		}
		fraction /= 10; //take it out
	}
	res = number + dfraction;
	return res;
}