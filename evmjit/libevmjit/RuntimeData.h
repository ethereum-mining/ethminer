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
		Gas,
		GasPrice,
		CallData,
		CallDataSize,
		Address,
		Caller,
		Origin,
		CallValue,
		CoinBase,
		Difficulty,
		GasLimit,
		Number,
		Timestamp,
		Code,
		CodeSize,

		SuicideDestAddress = Address,		///< Suicide balance destination address
		ReturnData 		   = CallData,		///< Return data pointer (set only in case of RETURN)
		ReturnDataSize 	   = CallDataSize,	///< Return data size (set only in case of RETURN)
	};

	int64_t 	gas = 0;
	int64_t 	gasPrice = 0;
	byte const* callData = nullptr;
	uint64_t 	callDataSize = 0;
	i256 		address;
	i256 		caller;
	i256 		origin;
	i256 		callValue;
	i256 		coinBase;
	i256 		difficulty;
	i256 		gasLimit;
	uint64_t 	number = 0;
	int64_t 	timestamp = 0;
	byte const* code = nullptr;
	uint64_t 	codeSize = 0;
};

/// VM Environment (ExtVM) opaque type
struct Env;

}
}
}
