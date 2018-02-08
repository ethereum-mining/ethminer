REM ethNipper.exe
REM Listing OpenCL devices (CLMiner).
REM [0]Platform: OpenCL 'NVIDIA CUDA'
REM   [-] no devices found
REM [1]Platform: OpenCL 'AMD Accelerated Parallel Processing'
REM   [0]Fiji       OpenCL C 2.0    RAM: 4096MB     CU: 56  WS: 256 1000Hz  (Advanced Micro Devices, Inc.)  ~25MH/s
REM   [1]Fiji       OpenCL C 2.0    RAM: 4096MB     CU: 56  WS: 256 1000Hz  (Advanced Micro Devices, Inc.)  ~25MH/s
REM   [2]Fiji       OpenCL C 2.0    RAM: 4096MB     CU: 56  WS: 256 1000Hz  (Advanced Micro Devices, Inc.)  ~25MH/s
REM   [3]Fiji       OpenCL C 2.0    RAM: 4096MB     CU: 56  WS: 256 1000Hz  (Advanced Micro Devices, Inc.)  ~25MH/s
REM   [4]Fiji       OpenCL C 2.0    RAM: 4096MB     CU: 56  WS: 256 1000Hz  (Advanced Micro Devices, Inc.)  ~25MH/s
REM   [5]Fiji       OpenCL C 2.0    RAM: 4096MB     CU: 56  WS: 256 1000Hz  (Advanced Micro Devices, Inc.)  ~25MH/s
REM Listing OpenCL devices (OCLMiner).
REM [2]Platform: OpenCL 'Altera SDK for OpenCL'
REM   [-] no devices found
REM ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
REM                                  0/1/2 -> Select AMD Accelerated Parallel Processing 
REM                                  |             1 Embeded default Stable Kernel / 2 Embeded default Unstable Kernel /3 sideload devicename.cl or kerner.cl as fallback
REM                                  |             |                5s for Nanopool 
REM                                  |             |                |     Report Hashrate to Nanopool
REM                                  |             |                |     |      Select Stratum Protokoll 1 for Hashrate Reporting to Nanopool
REM                                  |             |                |     |      |
start ethNipper.exe --opencl-platform 1 --cl-kernel 3 --farm-recheck 5000 -RH -SP 1 -HWMON -G -S eth-eu1.nanopool.org:9999 -O 0x-eth-address.workername/name@domain:x
