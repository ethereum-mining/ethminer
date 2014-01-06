#pragma once

#include "Common.h"
#include "RLP.h"

namespace eth
{

enum class AddressType
{
	Normal,
	Contract
};

class AddressState
{
public:
	AddressState(AddressType _type = AddressType::Normal): m_type(_type), m_balance(0), m_nonce(0) {}

	AddressType type() const { return m_type; }
	u256& balance() { return m_balance; }
	u256 const& balance() const { return m_balance; }
	u256& nonce() { return m_nonce; }
	u256 const& nonce() const { return m_nonce; }
	std::map<u256, u256>& memory() { assert(m_type == AddressType::Contract); return m_memory; }
	std::map<u256, u256> const& memory() const { assert(m_type == AddressType::Contract); return m_memory; }

	u256 memoryHash() const;

	std::string toString() const
	{
		if (m_type == AddressType::Normal)
			return rlpList(m_balance, toCompactBigEndianString(m_nonce));
		if (m_type == AddressType::Contract)
			return rlpList(m_balance, toCompactBigEndianString(m_nonce), toCompactBigEndianString(memoryHash()));
		return "";
	}

private:
	AddressType m_type;
	u256 m_balance;
	u256 m_nonce;
	u256Map m_memory;
};

}


