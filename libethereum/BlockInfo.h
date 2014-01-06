#pragma once

#include "Common.h"

namespace eth
{

struct BlockInfo
{
public:
	u256 hash;
	u256 parentHash;
	u256 sha256Uncles;
	u256 coinbaseAddress;
	u256 sha256Transactions;
	u256 difficulty;
	u256 timestamp;
	u256 nonce;
	u256 number;

	void populateAndVerify(bytesConstRef _block, u256 _number);
};

}


