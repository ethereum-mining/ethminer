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
#include <string>
#include <boost/exception/all.hpp>
#include <boost/throw_exception.hpp>
#include "CommonData.h"
#include "FixedHash.h"

namespace dev
{
// base class for all exceptions
struct Exception: virtual std::exception, virtual boost::exception { mutable std::string m_message; };

struct BadHexCharacter: virtual Exception {};
struct RLPException: virtual Exception {};
struct BadCast: virtual RLPException {};
struct BadRLP: virtual RLPException {};
struct NoNetworking: virtual Exception {};
struct NoUPnPDevice: virtual Exception {};
struct RootNotFound: virtual Exception {};
struct FileError: virtual Exception {};
struct InterfaceNotSupported: virtual Exception { public: InterfaceNotSupported(std::string _f): m_f("Interface " + _f + " not supported.") {} virtual const char* what() const noexcept { return m_f.c_str(); } private: std::string m_f; };

// error information to be added to exceptions
using errinfo_invalidSymbol = boost::error_info<struct tag_invalidSymbol, char>;
using errinfo_wrongAddress = boost::error_info<struct tag_address, std::string>;
using errinfo_comment = boost::error_info<struct tag_comment, std::string>;
using errinfo_required = boost::error_info<struct tag_required, bigint>;
using errinfo_got = boost::error_info<struct tag_got, bigint>;
using RequirementError = boost::tuple<errinfo_required, errinfo_got>;
}
