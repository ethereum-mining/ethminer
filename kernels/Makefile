SDIR=./src
BUILD=./bin

ASFLAGS=-I$(SDIR)
AS=clrxasm

.PHONY: all
all: ellesmere tonga baffin gfx901
	@echo "Built ethash kernel for all architectures.\n\n"

.PHONY: clean
clean:
	rm -f $(BUILD)/*.o $(BUILD)/*.bin

%.bin:
	$(eval ARCH:=$(shell echo "$*" | sed -n 's/\(^[A-Za-z0-9]*\).*/\1/p'))
	$(eval LWS:=$(shell echo "$*" | sed -n 's/.*lws\([0-9]\+\)/\1/p'))
	$(AS) $(SDIR)/GCN_ethash.isa -g $(ARCH) -Dworksize=$(LWS) $(ASFLAGS) \
	      -o $(BUILD)/ethash_$(ARCH)_lws$(LWS).bin

.PHONY: ellesmere
ellesmere: ellesmere_lws64.bin ellesmere_lws128.bin ellesmere_lws192.bin \
	       ellesmere_lws256.bin
		@echo "-------------\nBuilt Ellesmere kernels...\n-----------------"

.PHONY: tonga
tonga: tonga_lws64.bin tonga_lws128.bin tonga_lws192.bin \
	   tonga_lws256.bin
		@echo "----------------\nBuilt Tonga kernels...\n------------------"

.PHONY: baffin
baffin: baffin_lws64.bin baffin_lws128.bin baffin_lws192.bin \
	    baffin_lws256.bin
		@echo "----------------\nBuilt Baffin kernels...\n-----------------"

.PHONY: gfx901
gfx901: gfx901_lws64.bin gfx901_lws128.bin gfx901_lws192.bin \
	    gfx901_lws256.bin
		@echo "----------------\nBuilt gfx901 kernels...\n-----------------"
