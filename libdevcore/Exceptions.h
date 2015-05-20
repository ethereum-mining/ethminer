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
struct Exception: virtual std::exception, virtual boost::exception
{
	Exception(std::string _message = std::string()): m_message(std::move(_message)) {}
	const char* what() const noexcept override { return m_message.empty() ? std::exception::what() : m_message.c_str(); }

private:
	std::string m_message;
};

struct BadHexCharacter: virtual Exception {};
struct RLPException: virtual Exception {};
struct BadCast: virtual RLPException {};
struct BadRLP: virtual RLPException {};
struct OversizeRLP: virtual RLPException {};
struct UndersizeRLP: virtual RLPException {};
struct NoNetworking: virtual Exception {};
struct NoUPnPDevice: virtual Exception {};
struct RootNotFound: virtual Exception {};
struct BadRoot: virtual Exception {};
struct FileError: virtual Exception {};
struct Overflow: virtual Exception {};
struct InterfaceNotSupported: virtual Exception { public: InterfaceNotSupported(std::string _f): Exception("Interface " + _f + " not supported.") {} };
struct FailedInvariant: virtual Exception {};
struct ExternalFunctionFailure: virtual Exception { public: ExternalFunctionFailure(std::string _f): Exception("Function " + _f + "() failed.") {} };

// error information to be added to exceptions
using errinfo_invalidSymbol = boost::error_info<struct tag_invalidSymbol, char>;
using errinfo_wrongAddress = boost::error_info<struct tag_address, std::string>;
using errinfo_comment = boost::error_info<struct tag_comment, std::string>;
using errinfo_required = boost::error_info<struct tag_required, bigint>;
using errinfo_got = boost::error_info<struct tag_got, bigint>;
using errinfo_min = boost::error_info<struct tag_min, bigint>;
using errinfo_max = boost::error_info<struct tag_max, bigint>;
using RequirementError = boost::tuple<errinfo_required, errinfo_got>;
using errinfo_hash256 = boost::error_info<struct tag_hash, h256>;
using HashMismatchError = boost::tuple<errinfo_hash256, errinfo_hash256>;
}
