#pragma once

#include <array>
#include <map>
#include <unordered_map>
#include "Exceptions.h"
#include "AddressState.h"
#include "BlockInfo.h"
#include "RLP.h"
#include "Common.h"

namespace eth
{

struct Signature
{
	byte v;
	u256 r;
	u256 s;
};

using PrivateKey = u256;
using Address = u160;

// [ nonce, receiving_address, value, fee, [ data item 0, data item 1 ... data item n ], v, r, s ]
struct Transaction
{
	Transaction() {}
	Transaction(bytes const& _rlp);

	u256 nonce;
	Address receiveAddress;
	u256 value;
	u256 fee;
	u256s data;
	Signature vrs;

	Address sender() const;
	void sign(PrivateKey _priv);

	void fillStream(RLPStream& _s, bool _sig = true) const;
	bytes rlp(bool _sig = true) const { RLPStream s; fillStream(s, _sig); return s.out(); }
	std::string rlpString(bool _sig = true) const { RLPStream s; fillStream(s, _sig); return s.str(); }
	u256 sha256(bool _sig = true) const { RLPStream s; fillStream(s, _sig); return eth::sha256(s.out()); }
	bytes sha256Bytes(bool _sig = true) const { RLPStream s; fillStream(s, _sig); return eth::sha256Bytes(s.out()); }
};

class State
{
public:
	explicit State(Address _minerAddress);

	static void ensureCrypto();
	static std::mt19937_64& engine();

	bool verify(bytes const& _block);
	bool execute(bytes const& _rlp) { try { Transaction t(_rlp); execute(t, t.sender()); } catch (...) { return false; } }

private:
	struct MinerFeeAdder
	{
		~MinerFeeAdder() { state->addBalance(state->m_minerAddress, fee); }
		State* state;
		u256 fee;
	};

	void execute(Transaction const& _t, Address _sender);

	bool isNormalAddress(Address _address) const { auto it = m_current.find(_address); return it != m_current.end() && it->second.type() == AddressType::Normal; }
	bool isContractAddress(Address _address) const { auto it = m_current.find(_address); return it != m_current.end() && it->second.type() == AddressType::Contract; }

	u256 balance(Address _id) const { auto it = m_current.find(_id); return it == m_current.end() ? 0 : it->second.balance(); }
	void addBalance(Address _id, u256 _amount) { auto it = m_current.find(_id); if (it == m_current.end()) it->second.balance() = _amount; else it->second.balance() += _amount; }
	// bigint as we don't want any accidental problems with -ve numbers.
	bool subBalance(Address _id, bigint _amount) { auto it = m_current.find(_id); if (it == m_current.end() || (bigint)it->second.balance() < _amount) return false; it->second.balance() = (u256)((bigint)it->second.balance() - _amount); return true; }

	u256 contractMemory(Address _contract, u256 _memory) const;

	u256 transactionsFrom(Address _address) { auto it = m_current.find(_address); return it == m_current.end() ? 0 : it->second.nonce(); }

	void execute(Address _myAddress, Address _txSender, u256 _txValue, u256 _txFee, u256s const& _txData, u256* o_totalFee);

	std::map<Address, AddressState> m_current;
	BlockInfo m_previousBlock;
	BlockInfo m_currentBlock;

	Address m_minerAddress;

	static const u256 c_stepFee;
	static const u256 c_dataFee;
	static const u256 c_memoryFee;
	static const u256 c_extroFee;
	static const u256 c_cryptoFee;
	static const u256 c_newContractFee;
	static const u256 c_txFee;

	static std::mt19937_64* s_engine;
};

}


