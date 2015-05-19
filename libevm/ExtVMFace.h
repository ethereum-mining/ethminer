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
#include <libdevcore/SHA3.h>
#include <libevmcore/Instruction.h>
#include <libethcore/Common.h>
#include <libethcore/BlockInfo.h>

namespace dev
{
namespace eth
{

struct LogEntry
{
	LogEntry() {}
	LogEntry(RLP const& _r) { address = (Address)_r[0]; topics = _r[1].toVector<h256>(); data = _r[2].toBytes(); }
	LogEntry(Address const& _address, h256s const& _ts, bytes&& _d): address(_address), topics(_ts), data(std::move(_d)) {}

	void streamRLP(RLPStream& _s) const { _s.appendList(3) << address << topics << data; }

	LogBloom bloom() const
	{
		LogBloom ret;
		ret.shiftBloom<3>(sha3(address.ref()));
		for (auto t: topics)
			ret.shiftBloom<3>(sha3(t.ref()));
		return ret;
	}

	Address address;
	h256s topics;
	bytes data;
};

using LogEntries = std::vector<LogEntry>;

struct LocalisedLogEntry: public LogEntry
{
	LocalisedLogEntry() {}
	LocalisedLogEntry(LogEntry const& _le, unsigned _number, h256 _transactionHash = h256()): LogEntry(_le), number(_number), transactionHash(_transactionHash) {}

	unsigned number = 0;
	h256 transactionHash;
};

using LocalisedLogEntries = std::vector<LocalisedLogEntry>;

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
		logs += _s.logs;
		return *this;
	}

	void clear()
	{
		suicides.clear();
		logs.clear();
		refunds = 0;
	}
};

class ExtVMFace;
class VM;

using LastHashes = std::vector<h256>;

using OnOpFunc = std::function<void(uint64_t /*steps*/, Instruction /*instr*/, bigint /*newMemSize*/, bigint /*gasCost*/, VM*, ExtVMFace const*)>;

/**
 * @brief Interface and null implementation of the class for specifying VM externalities.
 */
class ExtVMFace
{
public:
	/// Null constructor.
	ExtVMFace() = default;

	/// Full constructor.
	ExtVMFace(Address _myAddress, Address _caller, Address _origin, u256 _value, u256 _gasPrice, bytesConstRef _data, bytes const& _code, BlockInfo const& _previousBlock, BlockInfo const& _currentBlock, LastHashes const& _lh, unsigned _depth);

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

	/// Does the account exist?
	virtual bool exists(Address) { return false; }

	/// Suicide the associated contract and give proceeds to the given address.
	virtual void suicide(Address) { sub.suicides.insert(myAddress); }

	/// Create a new (contract) account.
	virtual h160 create(u256, u256&, bytesConstRef, OnOpFunc const&) { return h160(); }

	/// Make a new message call.
	virtual bool call(Address, u256, bytesConstRef, u256&, bytesRef, OnOpFunc const&, Address, Address) { return false; }

	/// Revert any changes made (by any of the other calls).
	virtual void log(h256s&& _topics, bytesConstRef _data) { sub.logs.push_back(LogEntry(myAddress, std::move(_topics), _data.toBytes())); }

	/// Revert any changes made (by any of the other calls).
	virtual void revert() {}

	/// Hash of a block if within the last 256 blocks, or h256() otherwise.
	h256 blockhash(u256 _number) { return _number < currentBlock.number && _number >= (std::max<u256>(256, currentBlock.number) - 256) ? lastHashes[(unsigned)(currentBlock.number - 1 - _number)] : h256(); }

	/// Get the code at the given location in code ROM.
	byte getCode(u256 _n) const { return _n < code.size() ? code[(size_t)_n] : 0; }

	Address myAddress;			///< Address associated with executing code (a contract, or contract-to-be).
	Address caller;				///< Address which sent the message (either equal to origin or a contract).
	Address origin;				///< Original transactor.
	u256 value;					///< Value (in Wei) that was passed to this address.
	u256 gasPrice;				///< Price of gas (that we already paid).
	bytesConstRef data;			///< Current input data.
	bytes code;					///< Current code that is executing.
	LastHashes lastHashes;		///< Most recent 256 blocks' hashes.
	BlockInfo previousBlock;	///< The previous block's information.	TODO: PoC-8: REMOVE
	BlockInfo currentBlock;		///< The current block's information.
	SubState sub;				///< Sub-band VM state (suicides, refund counter, logs).
	unsigned depth = 0;			///< Depth of the present call.
};

}
}
