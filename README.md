Maetti's Fork (Ethereum) + Altera/Intel OpenCL(FPGA)

Forked version of https://github.com/ethereum-mining/ethminer

### Features

- OpenCL mining embedded original project kernels
- OpenCL mining Intel/Altera custom FPGA offline compiled binary kernels
- Sideload Custom OpenCL (devicename.cl) C99 kernels by GPU Type
- Stratum 

- Nvidia CUDA mining / disabled
- OpenCl CPU mining / disabled

custom kernels not included

Epoch #169 

 R9 290  Hawaii     RAM: 4GB CU: 40 WS: 256 1030Hz ~24MH/s 
 R9 290X Hawaii     RAM: 4GB CU: 44 WS: 256 1000Hz ~26MH/s 
 R9 290X Hawaii     RAM: 4GB CU: 44 WS: 256 1030Hz ~26MH/s 
 R9 290X Hawaii     RAM: 4GB CU: 44 WS: 256 1040Hz ~27MH/s 
 R9 290X Hawaii     RAM: 4GB CU: 44 WS: 256 1050Hz ~28MH/s *OC
 R9 380  Tonga      RAM: 4GB CU: 32 WS: 256 1030Hz ~16MH/s 
 R9 380  Tonga      RAM: 4GB CU: 32 WS: 256 1050Hz ~17MH/s 
 RX 480  Ellesmere  RAM: 8GB CU: 36 WS: 256 1266Hz ~17MH/s
 R9 Fury Fiji       RAM: 4GB CU: 56 WS: 256 1050Hz ~29MH/s 
 R9 Fury Fiji       RAM: 4GB CU: 56 WS: 256 1100Hz ~31MH/s *OC
 RX VEGA Vega       RAM: 8GB CU: 56 WS: 256 1000Hz ~33MH/s
 RX VEGA Vega       RAM: 8GB CU: 56 WS: 256 1150Hz ~41MH/s *OC
 FPGA    Stratix V  RAM: 8GB CU:  2 WS: 256  933Hz >50GH/s *DEBUG MODE
  