#pragma once

#include "Utils.h"


namespace dev
{
namespace eth
{
namespace jit
{
	
struct RuntimeData
{
	enum Index
	{
		Address,
		Caller,
		Origin,
		CallValue,
		GasPrice,
		CoinBase,
		Difficulty,
		GasLimit,
		CallData,
		Code,
		CodeSize,
		CallDataSize,
		Gas,
		BlockNumber,
		BlockTimestamp,

		SuicideDestAddress = Address,		///< Suicide balance destination address
		ReturnData 		   = CallData,		///< Return data pointer (set only in case of RETURN)
		ReturnDataSize 	   = CallDataSize,	///< Return data size (set only in case of RETURN)
	};

	i256 address;
	i256 caller;
	i256 origin;
	i256 callValue;
	i256 gasPrice;
	i256 coinBase;
	i256 difficulty;
	i256 gasLimit;
	byte const* callData = nullptr;
	byte const* code = nullptr;
	uint64_t codeSize = 0;
	uint64_t callDataSize = 0;
	int64_t gas = 0;
	uint64_t blockNumber = 0;
	uint64_t blockTimestamp = 0;
};

/// VM Environment (ExtVM) opaque type
struct Env;

}
}
}
