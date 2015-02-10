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
/** @file MixClient.h
* @author Yann yann@ethdev.com
* @author Arkadiy Paronyan arkadiy@ethdev.com
* @date 2015
* Ethereum IDE client.
*/

#pragma once

#include <vector>
#include <map>
#include <stdint.h>
#include <libdevcore/Common.h>
#include <libdevcrypto/Common.h>
#include <libevmcore/Instruction.h>
#include <libethereum/TransactionReceipt.h>

namespace dev
{
namespace mix
{

	/**
	* @brief Store information about a machine state.
	*/
	struct MachineState
	{
		uint64_t steps;
		dev::Address address;
		dev::u256 curPC;
		dev::eth::Instruction inst;
		dev::bigint newMemSize;
		dev::u256 gas;
		dev::u256s stack;
		dev::bytes memory;
		dev::bigint gasCost;
		std::map<dev::u256, dev::u256> storage;
		std::vector<unsigned> levels;
		unsigned codeIndex;
		unsigned dataIndex;
	};

	/**
	* @brief Store information about a machine states.
	*/
	struct ExecutionResult
	{
		ExecutionResult(): transactionIndex(std::numeric_limits<unsigned>::max()) {}

		std::vector<MachineState> machineStates;
		std::vector<bytes> transactionData;
		std::vector<bytes> executionCode;
		bytes returnValue;
		dev::Address address;
		dev::Address sender;
		dev::Address contractAddress;
		dev::u256 value;
		unsigned transactionIndex;

		bool isCall() const { return transactionIndex == std::numeric_limits<unsigned>::max(); }
	};

	using ExecutionResults = std::vector<ExecutionResult>;
}
}
