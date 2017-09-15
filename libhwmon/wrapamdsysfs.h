/*
* Wrapper for AMD SysFS on linux, using adapted code from amdcovc by matszpk
*
* By Philipp Andreas - github@smurfy.de
*/

#ifndef _WRAPAMDSYSFS_H_
#define _WRAPAMDSYSFS_H_

typedef struct {
	int sysfs_gpucount;
	int *card_sysfs_device_id;  /* map cardidx to filesystem card idx */
	int *sysfs_hwmon_id;        /* filesystem card idx to filesystem hwmon idx */
} wrap_amdsysfs_handle;

wrap_amdsysfs_handle * wrap_amdsysfs_create();
int wrap_amdsysfs_destory(wrap_amdsysfs_handle *sysfsh);

int wrap_amdsysfs_get_gpucount(wrap_amdsysfs_handle *sysfsh, int *gpucount);

int wrap_amdsysfs_get_tempC(wrap_amdsysfs_handle *sysfsh, int index, unsigned int *tempC);

int wrap_amdsysfs_get_fanpcnt(wrap_amdsysfs_handle *sysfsh, int index, unsigned int *fanpcnt);

#endif