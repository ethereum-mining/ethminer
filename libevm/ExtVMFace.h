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
/** @file ExtVMFace.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <set>
#include <functional>
#include <libdevcore/Common.h>
#include <libdevcore/CommonData.h>
#include <libdevcore/RLP.h>
#include <libdevcrypto/SHA3.h>
#include <libevmface/Instruction.h>
#include <libethcore/CommonEth.h>
#include <libethcore/BlockInfo.h>

namespace dev
{
namespace eth
{

using LogBloom = h512;

struct LogEntry
{
	LogEntry() {}
	LogEntry(RLP const& _r) { from = (Address)_r[0]; topics = (h256s)_r[1]; data = (bytes)_r[2]; }
	LogEntry(Address const& _f, h256s&& _ts, bytes&& _d): from(_f), topics(std::move(_ts)), data(std::move(_d)) {}

	void streamRLP(RLPStream& _s) const { _s.appendList(3) << from << topics << data; }

	LogBloom bloom() const
	{
		LogBloom ret;
		ret.shiftBloom<3, 32>(sha3(from.ref()));
		for (auto t: topics)
			ret.shiftBloom<3, 32>(sha3(t.ref()));
		return ret;
	}

	Address from;
	h256s topics;
	bytes data;
};

using LogEntries = std::vector<LogEntry>;

inline LogBloom bloom(LogEntries const& _logs)
{
	LogBloom ret;
	for (auto const& l: _logs)
		ret |= l.bloom();
	return ret;
}

struct SubState
{
	std::set<Address> suicides;	///< Any accounts that have suicided.
	LogEntries logs;			///< Any logs.
	u256 refunds;				///< Refund counter of SSTORE nonzero->zero.

	SubState& operator+=(SubState const& _s)
	{
		suicides += _s.suicides;
		refunds += _s.refunds;
		return *this;
	}
};

using OnOpFunc = std::function<void(uint64_t /*steps*/, Instruction /*instr*/, bigint /*newMemSize*/, bigint /*gasCost*/, void/*VM*/*, void/*ExtVM*/ const*)>;

/**
 * @brief Interface and null implementation of the class for specifying VM externalities.
 */
class ExtVMFace
{
public:
	/// Null constructor.
	ExtVMFace() = default;

	/// Full constructor.
	ExtVMFace(Address _myAddress, Address _caller, Address _origin, u256 _value, u256 _gasPrice, bytesConstRef _data, bytes _code, BlockInfo const& _previousBlock, BlockInfo const& _currentBlock, unsigned _depth);

	virtual ~ExtVMFace() = default;

	ExtVMFace(ExtVMFace const&) = delete;
	void operator=(ExtVMFace) = delete;

	/// Read storage location.
	virtual u256 store(u256) { return 0; }

	/// Write a value in storage.
	virtual void setStore(u256, u256) {}

	/// Read address's balance.
	virtual u256 balance(Address) { return 0; }

	/// Read address's code.
	virtual bytes const& codeAt(Address) { return NullBytes; }

	/// Subtract amount from account's balance.
	virtual void subBalance(u256) {}

	/// Determine account's TX count.
	virtual u256 txCount(Address) { return 0; }

	/// Suicide the associated contract and give proceeds to the given address.
	virtual void suicide(Address) { sub.suicides.insert(myAddress); }

	/// Create a new (contract) account.
	virtual h160 create(u256, u256*, bytesConstRef, OnOpFunc const&) { return h160(); }

	/// Make a new message call.
	virtual bool call(Address, u256, bytesConstRef, u256*, bytesRef, OnOpFunc const&, Address, Address) { return false; }

	/// Revert any changes made (by any of the other calls).
	virtual void log(h256s&& _topics, bytesConstRef _data) { sub.logs.push_back(LogEntry(myAddress, std::move(_topics), _data.toBytes())); }

	/// Revert any changes made (by any of the other calls).
	virtual void revert() {}

	/// Get the code at the given location in code ROM.
	byte getCode(u256 _n) const { return _n < code.size() ? code[(size_t)_n] : 0; }

	Address myAddress;			///< Address associated with executing code (a contract, or contract-to-be).
	Address caller;				///< Address which sent the message (either equal to origin or a contract).
	Address origin;				///< Original transactor.
	u256 value;					///< Value (in Wei) that was passed to this address.
	u256 gasPrice;				///< Price of gas (that we already paid).
	bytesConstRef data;			///< Current input data.
	bytes code;			///< Current code that is executing.
	BlockInfo previousBlock;	///< The previous block's information.
	BlockInfo currentBlock;		///< The current block's information.
	SubState sub;				///< Sub-band VM state (suicides, refund counter, logs).
	unsigned depth;				///< Depth of the present call.
};

}
}
