#include<sys/stat.h>
#include<stdlib.h>
#include<stdio.h>
#include<stdarg.h>
#include<CL/cl.h>
#include<platform/icd_test_log.h>

#define APP_LOG_FILE  "icd_test_app_log.txt"
#define STUB_LOG_FILE "icd_test_stub_log.txt"

static FILE *app_log_file;
static FILE *stub_log_file;

int test_icd_initialize_app_log(void)
{
    app_log_file = fopen(APP_LOG_FILE, "w");
    if (!app_log_file) {
		printf("Unable to open file %s\n", APP_LOG_FILE);
        return -1;
    }
}

void test_icd_close_app_log(void)
{
    fclose(app_log_file);
}

void test_icd_app_log(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vfprintf(app_log_file, format, args);
    va_end(args);
}

int test_icd_initialize_stub_log(void)
{
   	stub_log_file = fopen(STUB_LOG_FILE, "w");
    if (!stub_log_file) {
		printf("Unable to open file %s\n", STUB_LOG_FILE);
        return -1;
    }
}

void test_icd_close_stub_log(void)
{
    fclose(stub_log_file);
}

void test_icd_stub_log(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vfprintf(stub_log_file, format, args);
    va_end(args);
}

static char *test_icd_get_log(const char *filename)
{
    struct stat statbuf;
    FILE *fp;
    char *source = NULL;

    fp = fopen(filename, "rb");

    if (fp) {
        size_t fsize = 0;
        stat(filename, &statbuf);
        fsize = statbuf.st_size;
        source = (char *)malloc(fsize+1); // +1 for NULL terminator
        if (source) {
            if (fsize) {
                if (fread(source, fsize, 1, fp) != 1) {
                    free(source);
                    source = NULL;
                } else {
                    source[fsize] = '\0';
                }
            } else {
                // Don't fail when fsize = 0, just return empty string
                source[fsize] = '\0';
            }
        }
        fclose(fp);
    }

    return source;
}

char *test_icd_get_app_log(void)
{
    return test_icd_get_log(APP_LOG_FILE);
}

char *test_icd_get_stub_log(void)
{
    return test_icd_get_log(STUB_LOG_FILE);
}
