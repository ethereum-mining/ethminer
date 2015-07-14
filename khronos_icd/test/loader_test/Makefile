CC := gcc
CFLAGS := -I ../inc -I ../../ -g -O0

OUTDIR := ../../bin

${OUTDIR}/icd_loader_test: main.c callbacks.c icd_test_match.c

${OUTDIR}/icd_loader_test: test_buffer_object.c test_clgl.c test_cl_runtime.c test_create_calls.c test_image_objects.c

${OUTDIR}/icd_loader_test: test_kernel.c test_platforms.c test_program_objects.c test_sampler_objects.c
	${CC} ${CFLAGS} ${OUTDIR}/libOpenCL.so -o $@ $^ ${OUTDIR}/libIcdLog.so

.PHONY: clean

clean:
	rm -f ${OUTDIR}/icd_loader_test
