SDIR=./isa
BUILD=./bin

ASFLAGS=-I$(SDIR)
AS=clrxasm

.PHONY: all
all: ellesmere tonga baffin gfx900 gfx901 gfx906
	@echo "Built ethash kernel for Ellesmere, Tonga, Baffin, Gfx900 and Gfx901 architectures."

.PHONY: clean
clean:
	rm -f $(BUILD)/*.o $(BUILD)/*.bin

%.bin:
	$(eval ARCH:=$(shell echo "$*" | sed -n 's/\(^[A-Za-z0-9]*\).*/\1/p'))
	$(eval LWS:=$(shell echo "$*" | sed -n 's/.*lws\([0-9]\+\)/\1/p'))
	$(AS) $(SDIR)/GCN_ethash.isa -g $(ARCH) -DFAST_EXIT -Dworksize=$(LWS) $(ASFLAGS) -o $(BUILD)/ethash_$(ARCH)_lws$(LWS)_exit.bin
	$(AS) $(SDIR)/GCN_ethash.isa -g $(ARCH) -Dworksize=$(LWS) $(ASFLAGS) -o $(BUILD)/ethash_$(ARCH)_lws$(LWS).bin

.PHONY: ellesmere
ellesmere: ellesmere_lws64.bin ellesmere_lws128.bin ellesmere_lws256.bin
		@echo "Built Ellesmere kernels..."

.PHONY: tonga
tonga: tonga_lws64.bin tonga_lws128.bin tonga_lws256.bin
		@echo "Built Tonga kernels..."

.PHONY: baffin
baffin: baffin_lws64.bin baffin_lws128.bin baffin_lws256.bin
		@echo "Built Baffin kernels..."

.PHONY: gfx900
gfx900: gfx900_lws64.bin gfx900_lws128.bin gfx900_lws256.bin
		@echo "Built gfx900 kernels..."

.PHONY: gfx901
gfx901: gfx901_lws64.bin gfx901_lws128.bin gfx901_lws256.bin
		@echo "Built gfx901 kernels..."

.PHONY: gfx906
gfx906: gfx906_lws64.bin gfx906_lws128.bin gfx906_lws256.bin
		@echo "Built gfx906 kernels..."
