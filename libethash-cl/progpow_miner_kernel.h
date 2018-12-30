/*
This file is part of ethminer.

ethminer is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ethminer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ethminer.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

/*
   cl_progpow_miner_kernel() and sizeof_cl_progpow_miner_kernel()
   are generated from progpow_miner_kernel.cu
   using cmake ../libethash-cl/bin2h.cmake
*/
extern const char* cl_progpow_miner_kernel(void);
extern size_t sizeof_cl_progpow_miner_kernel(void);
