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

#include <memory>
#include <libdevcore/Exceptions.h>
#include "ExtVMFace.h"

namespace dev
{
namespace eth
{

struct VMException: virtual Exception {};
struct StepsDone: virtual VMException {};
struct BreakPointHit: virtual VMException {};
struct BadInstruction: virtual VMException {};
struct BadJumpDestination: virtual VMException {};
struct OutOfGas: virtual VMException {};
class StackTooSmall: virtual public VMException { public: StackTooSmall(u256 _req, u256 _got): req(_req), got(_got) {} u256 req; u256 got; };

// Convert from a 256-bit integer stack/memory entry into a 160-bit Address hash.
// Currently we just pull out the right (low-order in BE) 160-bits.
inline Address asAddress(u256 _item)
{
	return right160(h256(_item));
}

inline u256 fromAddress(Address _a)
{
	return (u160)_a;
	//	h256 ret;
	//	memcpy(&ret, &_a, sizeof(_a));
	//	return ret;
}

/**
 */
class VMFace
{
public:
	/// Construct VM object.
	explicit VMFace(u256 _gas = 0): m_gas(_gas) {}

	virtual ~VMFace() = default;

	VMFace(VMFace const&) = delete;
	void operator=(VMFace const&) = delete;

	virtual void reset(u256 _gas = 0) noexcept { m_gas = _gas; }

	virtual bytesConstRef go(ExtVMFace& _ext, OnOpFunc const& _onOp = {}, uint64_t _steps = (uint64_t)-1) = 0;

	u256 gas() const { return m_gas; }

	enum Kind: bool { Interpreter, JIT };

	static std::unique_ptr<VMFace> create(Kind, u256 _gas = 0);

protected:
	u256 m_gas = 0;
};

}
}
