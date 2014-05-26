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
/** @file Exceptions.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <exception>
#include "CommonIO.h"
#include "CommonData.h"
#include "FixedHash.h"

namespace eth
{

class Exception: public std::exception
{
public:
	virtual std::string description() const { return typeid(*this).name(); }
	virtual char const* what() const noexcept { return typeid(*this).name(); }
};

class BadHexCharacter: public Exception {};

class RLPException: public Exception {};
class BadCast: public RLPException {};
class BadRLP: public RLPException {};
class NoNetworking: public Exception {};
class NoUPnPDevice: public Exception {};
class RootNotFound: public Exception {};

}
