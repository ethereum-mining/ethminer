CC := gcc
CFLAGS := -I ../inc -I ../../ -fPIC -g -O0

OUTDIR := ../../bin

${OUTDIR}/libOpenCLDriverStub.so: cl.c cl_ext.c cl_gl.c icd.c
	${CC} ${CFLAGS} -shared -Wl,-soname,$@ \
		-Wl,--version-script,icd_driver_exports.map \
		-o $@ $^ ${OUTDIR}/libIcdLog.so

.PHONY: clean

clean:
	rm -f ${OUTDIR}/libOpenCLDriverStub.so
