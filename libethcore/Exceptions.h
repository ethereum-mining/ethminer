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

#include <libdevcore/Exceptions.h>

namespace dev
{
namespace eth
{

// information to add to exceptions
using errinfo_name = boost::error_info<struct tag_field, std::string>;
using errinfo_field = boost::error_info<struct tag_field, int>;
using errinfo_data = boost::error_info<struct tag_data, std::string>;
using BadFieldError = boost::tuple<errinfo_field, errinfo_data>;

struct DatabaseAlreadyOpen: virtual dev::Exception {};
struct NotEnoughCash: virtual dev::Exception {};
struct GasPriceTooLow: virtual dev::Exception {};
struct BlockGasLimitReached: virtual dev::Exception {};
struct NoSuchContract: virtual dev::Exception {};
struct ContractAddressCollision: virtual dev::Exception {};
struct FeeTooSmall: virtual dev::Exception {};
struct TooMuchGasUsed: virtual dev::Exception {};
struct ExtraDataTooBig: virtual dev::Exception {};
struct InvalidSignature: virtual dev::Exception {};
class InvalidBlockFormat: virtual public dev::Exception { public: InvalidBlockFormat(int _f, bytesConstRef _d): m_f(_f), m_d(_d.toBytes()) {} int m_f; bytes m_d; virtual const char* what() const noexcept; };
struct InvalidUnclesHash: virtual dev::Exception {};
struct InvalidUncle: virtual dev::Exception {};
struct UncleTooOld: virtual dev::Exception {};
class UncleInChain: virtual public dev::Exception { public: UncleInChain(h256Set _uncles, h256 _block): m_uncles(_uncles), m_block(_block) {} h256Set m_uncles; h256 m_block; virtual const char* what() const noexcept; };
struct DuplicateUncleNonce: virtual dev::Exception {};
struct InvalidStateRoot: virtual dev::Exception {};
class InvalidTransactionsHash: virtual public dev::Exception { public: InvalidTransactionsHash(h256 _head, h256 _real): m_head(_head), m_real(_real) {} h256 m_head; h256 m_real; virtual const char* what() const noexcept; };
struct InvalidTransaction: virtual dev::Exception {};
struct InvalidDifficulty: virtual dev::Exception {};
class InvalidGasLimit: virtual public dev::Exception { public: InvalidGasLimit(u256 _provided = 0, u256 _valid = 0): provided(_provided), valid(_valid) {} u256 provided; u256 valid; virtual const char* what() const noexcept; };
class InvalidMinGasPrice: virtual public dev::Exception { public: InvalidMinGasPrice(u256 _provided = 0, u256 _limit = 0): provided(_provided), limit(_limit) {} u256 provided; u256 limit; virtual const char* what() const noexcept; };
struct InvalidTransactionGasUsed: virtual dev::Exception {};
struct InvalidTransactionsStateRoot: virtual dev::Exception {};
struct InvalidReceiptsStateRoot: virtual dev::Exception {};
struct InvalidTimestamp: virtual dev::Exception {};
struct InvalidLogBloom: virtual dev::Exception {};
class InvalidNonce: virtual public dev::Exception { public: InvalidNonce(u256 _required = 0, u256 _candidate = 0): required(_required), candidate(_candidate) {} u256 required; u256 candidate; virtual const char* what() const noexcept; };
class InvalidBlockNonce: virtual public dev::Exception { public: InvalidBlockNonce(h256 _h = h256(), h256 _n = h256(), u256 _d = 0): h(_h), n(_n), d(_d) {} h256 h; h256 n; u256 d; virtual const char* what() const noexcept; };
struct InvalidParentHash: virtual dev::Exception {};
struct InvalidNumber: virtual dev::Exception {};
struct InvalidContractAddress: virtual public dev::Exception {};

}
}
