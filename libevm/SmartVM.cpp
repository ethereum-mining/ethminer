/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/

// SmartVM is only available if EVM JIT is enabled
#if ETH_EVMJIT

#include "SmartVM.h"
#include <unordered_map>
#include <libdevcore/Log.h>
#include <libdevcrypto/SHA3.h>
#include <evmjit/JIT.h>
#include <evmjit/libevmjit-cpp/Utils.h>
#include "VMFactory.h"

namespace dev
{
namespace eth
{
namespace
{
	using HitMap = std::unordered_map<h256, uint64_t>;

	HitMap& getHitMap()
	{
		static HitMap s_hitMap;
		return s_hitMap;
	}
}

bytesConstRef SmartVM::go(u256& io_gas, ExtVMFace& _ext, OnOpFunc const& _onOp, uint64_t _steps)
{
	auto codeHash = sha3(_ext.code);
	auto vmKind = VMKind::Interpreter; // default VM

	// Jitted EVM code already in memory?
	if (evmjit::JIT::isCodeReady(eth2llvm(codeHash)))
	{
		cnote << "Jitted";
		vmKind = VMKind::JIT;
	}
	else
	{
		// Check EVM code hit count
		static const uint64_t c_hitTreshold = 1;
		auto& hits = getHitMap()[codeHash];
		++hits;
		if (hits > c_hitTreshold)
		{
			cnote << "JIT selected";
			vmKind = VMKind::JIT;
		}
	}

	// TODO: Selected VM must be kept only because it returns reference to its internal memory.
	//       VM implementations should be stateless, without escaping memory reference.
	m_selectedVM = VMFactory::create(vmKind);
	return m_selectedVM->go(io_gas, _ext, _onOp, _steps);
}

}
}

#endif
