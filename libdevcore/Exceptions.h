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
#include <boost/exception/all.hpp>
#include <boost/throw_exception.hpp>
#include "CommonIO.h"
#include "CommonData.h"
#include "FixedHash.h"

namespace dev
{

class Exception: public std::exception
{
public:
	virtual std::string description() const { return typeid(*this).name(); }
	virtual char const* what() const noexcept { return typeid(*this).name(); }
};

// As an exemplar case I only restructure BadRLP, if I would restrucutre everything the above Exception class
// can be replaced completely.

struct BException: virtual boost::exception, virtual std::exception {};

// there is no need to derive from any other class then BException just to add more information.
// This can be done dynamically during runtime.

struct BadRLP: virtual BException {};


class BadHexCharacter: public Exception {};
class RLPException: public BException {};
class BadCast: public RLPException {};
class NoNetworking: public Exception {};
class NoUPnPDevice: public Exception {};
class RootNotFound: public Exception {};

}
