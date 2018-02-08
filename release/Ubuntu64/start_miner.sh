# ./ethNipper
# Listing OpenCL devices (CLMiner).
# [0]Platform: OpenCL 'NVIDIA CUDA'
#   [-] no devices found
# [1]Platform: OpenCL 'AMD Accelerated Parallel Processing'
#   [0]Fiji       OpenCL C 2.0    RAM: 4096MB     CU: 56  WS: 256 1000Hz  (Advanced Micro Devices, Inc.)  ~25MH/s
#   [1]Fiji       OpenCL C 2.0    RAM: 4096MB     CU: 56  WS: 256 1000Hz  (Advanced Micro Devices, Inc.)  ~25MH/s
#   [2]Fiji       OpenCL C 2.0    RAM: 4096MB     CU: 56  WS: 256 1000Hz  (Advanced Micro Devices, Inc.)  ~25MH/s
#   [3]Fiji       OpenCL C 2.0    RAM: 4096MB     CU: 56  WS: 256 1000Hz  (Advanced Micro Devices, Inc.)  ~25MH/s
#   [4]Fiji       OpenCL C 2.0    RAM: 4096MB     CU: 56  WS: 256 1000Hz  (Advanced Micro Devices, Inc.)  ~25MH/s
#   [5]Fiji       OpenCL C 2.0    RAM: 4096MB     CU: 56  WS: 256 1000Hz  (Advanced Micro Devices, Inc.)  ~25MH/s
# Listing OpenCL devices (OCLMiner).
# [2]Platform: OpenCL 'Altera SDK for OpenCL'
#   [-] no devices found
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                             0/1/2 -> Select AMD Accelerated Parallel Processing 
#                             |             1 Embeded default Stable Kernel / 2 Embeded default Unstable Kernel /3 sideload devicename.cl or kerner.cl as fallback
#                             |             |                5s for Nanopool 
#                             |             |                |     Report Hashrate to Nanopool
#                             |             |                |     |      Select Stratum Protokoll 1 for Hashrate Reporting to Nanopool
#                             |             |                |     |      |
./ethNipper --opencl-platform 0 --cl-kernel 3 --farm-recheck 5000 -RH -SP 1 -HWMON -G -S eth-eu1.nanopool.org:9999 -O 0x-eth-address.workername/name@domain:x


