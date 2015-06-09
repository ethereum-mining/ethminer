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
/** @file Common.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Ethereum-specific data structures & algorithms.
 */

#pragma once

#include <string>
#include <functional>
#include <libdevcore/Common.h>
#include <libdevcore/FixedHash.h>
#include <libdevcrypto/Common.h>

namespace dev
{
namespace eth
{

/// Current protocol version.
extern const unsigned c_protocolVersion;

/// Current minor protocol version.
extern const unsigned c_minorProtocolVersion;

/// Current database version.
extern const unsigned c_databaseVersion;

/// The network id.
enum class Network
{
	Olympic = 0,
	Frontier = 1
};
extern const Network c_network;

/// User-friendly string representation of the amount _b in wei.
std::string formatBalance(bigint const& _b);

/// Get information concerning the currency denominations.
std::vector<std::pair<u256, std::string>> const& units();

/// The log bloom's size (2048-bit).
using LogBloom = h2048;

/// Many log blooms.
using LogBlooms = std::vector<LogBloom>;

template <size_t n> inline u256 exp10()
{
	return exp10<n - 1>() * u256(10);
}

template <> inline u256 exp10<0>()
{
	return u256(1);
}

// The various denominations; here for ease of use where needed within code.
static const u256 ether = exp10<18>();
static const u256 finney = exp10<15>();
static const u256 szabo = exp10<12>();
static const u256 wei = exp10<0>();

using Nonce = h64;

using BlockNumber = unsigned;

static const BlockNumber LatestBlock = (BlockNumber)-2;
static const BlockNumber PendingBlock = (BlockNumber)-1;

enum class RelativeBlock: BlockNumber
{
	Latest = LatestBlock,
	Pending = PendingBlock
};

enum class ImportResult
{
	Success = 0,
	UnknownParent,
	FutureTime,
	AlreadyInChain,
	AlreadyKnown,
	Malformed,
	BadChain
};

struct ImportRequirements
{
	using value = unsigned;
	enum
	{
		ValidNonce = 1, ///< Validate nonce
		DontHave = 2, ///< Avoid old blocks
		CheckUncles = 4, ///< Check uncle nonces
		Default = ValidNonce | DontHave | CheckUncles
	};
};

/// Super-duper signal mechanism. TODO: replace with somthing a bit heavier weight.
class Signal
{
public:
	class HandlerAux
	{
		friend class Signal;

	public:
		~HandlerAux() { if (m_s) m_s->m_fire.erase(m_i); m_s = nullptr; }

	private:
		HandlerAux(unsigned _i, Signal* _s): m_i(_i), m_s(_s) {}

		unsigned m_i = 0;
		Signal* m_s = nullptr;
	};

	using Callback = std::function<void()>;

	std::shared_ptr<HandlerAux> add(Callback const& _h) { auto n = m_fire.empty() ? 0 : (m_fire.rbegin()->first + 1); m_fire[n] = _h; return std::shared_ptr<HandlerAux>(new HandlerAux(n, this)); }

	void operator()() { for (auto const& f: m_fire) f.second(); }

private:
	std::map<unsigned, Callback> m_fire;
};

using Handler = std::shared_ptr<Signal::HandlerAux>;

struct TransactionSkeleton
{
	bool creation = false;
	Address from;
	Address to;
	u256 value;
	bytes data;
	u256 gas = UndefinedU256;
	u256 gasPrice = UndefinedU256;
};

void badBlockHeader(bytesConstRef _header, std::string const& _err);
inline void badBlockHeader(bytes const& _header, std::string const& _err) { badBlockHeader(&_header, _err); }
void badBlock(bytesConstRef _header, std::string const& _err);
inline void badBlock(bytes const& _header, std::string const& _err) { badBlock(&_header, _err); }

}
}
