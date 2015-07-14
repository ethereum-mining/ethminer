
.PHONY: all platform driver_stub loader_test clean

all: platform driver_stub loader_test

platform:
	${MAKE} -C platform

driver_stub:
	${MAKE} -C driver_stub

loader_test:
	${MAKE} -C loader_test

clean:
	${MAKE} -C platform clean
	${MAKE} -C driver_stub clean
	${MAKE} -C loader_test clean
