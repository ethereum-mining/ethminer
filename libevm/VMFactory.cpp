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

#include "VMFactory.h"
#include <libdevcore/Assertions.h>
#include "VM.h"

#if ETH_EVMJIT
#include <evmjit/libevmjit-cpp/JitVM.h>
#endif

namespace dev
{
namespace eth
{
namespace
{
	VMKind g_kind = VMKind::Interpreter;
}

void VMFactory::setKind(VMKind _kind)
{
	g_kind = _kind;
}

std::unique_ptr<VMFace> VMFactory::create()
{
#if ETH_EVMJIT
	return std::unique_ptr<VMFace>(g_kind == VMKind::JIT ? static_cast<VMFace*>(new JitVM) : static_cast<VMFace*>(new VM));
#else
	asserts(g_kind == VMKind::Interpreter && "JIT disabled in build configuration");
	return std::unique_ptr<VMFace>(new VM);
#endif
}

}
}
