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
#include "Common.h"

namespace dev
{
namespace eth
{

// information to add to exceptions
using errinfo_name = boost::error_info<struct tag_field, std::string>;
using errinfo_field = boost::error_info<struct tag_field, int>;
using errinfo_data = boost::error_info<struct tag_data, std::string>;
using errinfo_nonce = boost::error_info<struct tag_nonce, h64>;
using errinfo_difficulty = boost::error_info<struct tag_difficulty, u256>;
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
class InvalidBlockFormat: virtual public dev::Exception {};
struct InvalidUnclesHash: virtual dev::Exception {};
struct InvalidUncle: virtual dev::Exception {};
struct TooManyUncles: virtual dev::Exception {};
struct UncleTooOld: virtual dev::Exception {};
class UncleInChain: virtual public dev::Exception {};
struct DuplicateUncleNonce: virtual dev::Exception {};
struct InvalidStateRoot: virtual dev::Exception {};
struct InvalidGasUsed: virtual dev::Exception {};
class InvalidTransactionsHash: virtual public dev::Exception {};
struct InvalidTransaction: virtual dev::Exception {};
struct InvalidDifficulty: virtual dev::Exception {};
struct InvalidSeedHash: virtual dev::Exception {};
class InvalidGasLimit: virtual public dev::Exception {};
struct InvalidTransactionGasUsed: virtual dev::Exception {};
struct InvalidTransactionsStateRoot: virtual dev::Exception {};
struct InvalidReceiptsStateRoot: virtual dev::Exception {};
struct InvalidTimestamp: virtual dev::Exception {};
struct InvalidLogBloom: virtual dev::Exception {};
class InvalidNonce: virtual public dev::Exception {};
class InvalidBlockHeaderItemCount: virtual public dev::Exception {};
class InvalidBlockNonce: virtual public dev::Exception {};
struct InvalidParentHash: virtual dev::Exception {};
struct InvalidNumber: virtual dev::Exception {};
struct InvalidContractAddress: virtual public dev::Exception {};

}
}
