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
/** @file Precompiled.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <unordered_map>
#include <functional>
#include <libdevcore/CommonData.h>

namespace dev
{
namespace eth
{

/// Information structure regarding an account that is precompiled (i.e. 1, 2, 3).
struct PrecompiledAddress
{
	std::function<bigint(bytesConstRef)> gas;
	std::function<void(bytesConstRef, bytesRef)> exec;
};

/// Info on precompiled contract accounts baked into the protocol.
std::unordered_map<unsigned, PrecompiledAddress> const& precompiled();

}
}
