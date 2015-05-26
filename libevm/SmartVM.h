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
#pragma once

#include "VMFace.h"

namespace dev
{
namespace eth
{

/// Smart VM proxy.
///
/// This class is a strategy pattern implementation for VM. For every EVM code
/// execution request it tries to select the best VM implementation (Interpreter or JIT)
/// by analyzing available information like: code size, hit count, JIT status, etc.
class SmartVM: public VMFace
{
public:
	SmartVM(u256 const& _gas): m_gas(_gas) {}

	virtual bytesConstRef go(ExtVMFace& _ext, OnOpFunc const& _onOp = {}, uint64_t _steps = (uint64_t)-1) override final;

	void reset(u256 const& _gas = 0) noexcept override { m_gas = _gas; }
	u256 gas() const noexcept override { return (u256)m_gas; }

private:
	std::unique_ptr<VMFace> m_selectedVM;
	bigint m_gas;
};

}
}
