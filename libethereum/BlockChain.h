#pragma once

#include "Common.h"

namespace eth
{

/**
 * @brief Models the blockchain database.
 */
class BlockChain
{
public:
	BlockChain();
	~BlockChain();
	
	void import(bytes const& _block) {}

private:
};

}


