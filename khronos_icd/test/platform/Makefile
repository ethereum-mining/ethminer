# Set this if system does not have OpenCL headers in standard include directory
CL_HEADER_PATH := ../../

.PHONY: clean

CFLAGS := -I${CL_HEADER_PATH}
CFLAGS += -fPIC

OUTDIR := ../../bin

${OUTDIR}/libIcdLog.so: icd_test_log.c
	${CC} ${CFLAGS} -shared -Wl,-soname,libIcdLog.so -o $@ $^

clean:
	rm -f ${OUTDIR}/libIcdLog.so
