#ifndef _ICD_TEST_LOG_H_
#define _ICD_TEST_LOG_H_

#if defined (_WIN32)
#define DllExport   __declspec( dllexport ) 
#else
#define DllExport
#endif

DllExport int test_icd_initialize_app_log(void);
DllExport void test_icd_app_log(const char *format, ...);
DllExport void test_icd_close_app_log(void);
DllExport char *test_icd_get_stub_log(void);

DllExport int test_icd_initialize_stub_log(void);
DllExport void test_icd_stub_log(const char *format, ...);
DllExport void test_icd_close_stub_log(void);
DllExport char *test_icd_get_app_log(void);

#endif /* _ICD_TEST_LOG_H_ */
