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
/** @file Transaction.cpp
 * @author Dmitrii Khokhlov <winsvega@mail.ru>
 * @date 2015
 * Transaaction test functions.
 */

#include "test/TestHelper.h"
#include <libethcore/Exceptions.h>
#include <libevm/VMFace.h>
#include <libethcore/Common.h>

using namespace dev;
using namespace eth;

BOOST_AUTO_TEST_SUITE(libethereum)

BOOST_AUTO_TEST_CASE(TransactionGasRequired)
{
	Transaction tr(fromHex("0xf86d800182521c94095e7baea6a6c7c4c2dfeb977efac326af552d870a8e0358ac39584bc98a7c979f984b031ba048b55bfa915ac795c431978d8a6a992b628d557da5ff759b307d495a36649353a0efffd310ac743f371de3b9f7f9cb56c0b28ad43601b4ab949f53faa07bd2c804"), CheckTransaction::None);
	BOOST_CHECK_MESSAGE(tr.gasRequired() == 21952, "Transaction::GasRequired() has changed!");
}

BOOST_AUTO_TEST_CASE(TransactionConstructor)
{
	bool wasException = false;
	try
	{
		Transaction(fromHex("0xf86d800182521c94095e7baea6a6c7c4c2dfeb977efac326af552d870a8e0358ac39584bc98a7c979f984b031ba048b55bfa915ac795c431978d8a6a992b628d557da5ff759b307d495a36649353a0efffd310ac743f371de3b9f7f9cb56c0b28ad43601b4ab949f53faa07bd2c804"), CheckTransaction::Everything);
	}
	catch (OutOfGasIntrinsic)
	{
		wasException = true;
	}
	catch (Exception)
	{
		BOOST_ERROR("Exception thrown but expected OutOfGasIntrinsic instead");
	}

	BOOST_CHECK_MESSAGE(wasException, "Expected OutOfGasIntrinsic exception to be thrown at TransactionConstructor test");
}

BOOST_AUTO_TEST_CASE(ExecutionResultOutput)
{
	std::stringstream buffer;
	ExecutionResult exRes;

	exRes.gasUsed = u256("12345");
	exRes.newAddress = Address("a94f5374fce5edbc8e2a8697c15331677e6ebf0b");
	exRes.output = fromHex("001122334455");

	buffer << exRes;
	BOOST_CHECK_MESSAGE(buffer.str() == "{12345, a94f5374fce5edbc8e2a8697c15331677e6ebf0b, 001122334455}", "Error ExecutionResultOutput");
}

BOOST_AUTO_TEST_CASE(transactionExceptionOutput)
{
	std::stringstream buffer;
	buffer << TransactionException::BadInstruction;
	BOOST_CHECK_MESSAGE(buffer.str() == "BadInstruction", "Error output TransactionException::BadInstruction");
	buffer.str(std::string());

	buffer << TransactionException::None;
	BOOST_CHECK_MESSAGE(buffer.str() == "None", "Error output TransactionException::None");
	buffer.str(std::string());

	buffer << TransactionException::BadRLP;
	BOOST_CHECK_MESSAGE(buffer.str() == "BadRLP", "Error output TransactionException::BadRLP");
	buffer.str(std::string());

	buffer << TransactionException::InvalidFormat;
	BOOST_CHECK_MESSAGE(buffer.str() == "InvalidFormat", "Error output TransactionException::InvalidFormat");
	buffer.str(std::string());

	buffer << TransactionException::OutOfGasIntrinsic;
	BOOST_CHECK_MESSAGE(buffer.str() == "OutOfGasIntrinsic", "Error output TransactionException::OutOfGasIntrinsic");
	buffer.str(std::string());

	buffer << TransactionException::InvalidSignature;
	BOOST_CHECK_MESSAGE(buffer.str() == "InvalidSignature", "Error output TransactionException::InvalidSignature");
	buffer.str(std::string());

	buffer << TransactionException::InvalidNonce;
	BOOST_CHECK_MESSAGE(buffer.str() == "InvalidNonce", "Error output TransactionException::InvalidNonce");
	buffer.str(std::string());

	buffer << TransactionException::NotEnoughCash;
	BOOST_CHECK_MESSAGE(buffer.str() == "NotEnoughCash", "Error output TransactionException::NotEnoughCash");
	buffer.str(std::string());

	buffer << TransactionException::OutOfGasBase;
	BOOST_CHECK_MESSAGE(buffer.str() == "OutOfGasBase", "Error output TransactionException::OutOfGasBase");
	buffer.str(std::string());

	buffer << TransactionException::BlockGasLimitReached;
	BOOST_CHECK_MESSAGE(buffer.str() == "BlockGasLimitReached", "Error output TransactionException::BlockGasLimitReached");
	buffer.str(std::string());

	buffer << TransactionException::BadInstruction;
	BOOST_CHECK_MESSAGE(buffer.str() == "BadInstruction", "Error output TransactionException::BadInstruction");
	buffer.str(std::string());

	buffer << TransactionException::BadJumpDestination;
	BOOST_CHECK_MESSAGE(buffer.str() == "BadJumpDestination", "Error output TransactionException::BadJumpDestination");
	buffer.str(std::string());

	buffer << TransactionException::OutOfGas;
	BOOST_CHECK_MESSAGE(buffer.str() == "OutOfGas", "Error output TransactionException::OutOfGas");
	buffer.str(std::string());

	buffer << TransactionException::OutOfStack;
	BOOST_CHECK_MESSAGE(buffer.str() == "OutOfStack", "Error output TransactionException::OutOfStack");
	buffer.str(std::string());

	buffer << TransactionException::StackUnderflow;
	BOOST_CHECK_MESSAGE(buffer.str() == "StackUnderflow", "Error output TransactionException::StackUnderflow");
	buffer.str(std::string());

	buffer << TransactionException(-1);
	BOOST_CHECK_MESSAGE(buffer.str() == "Unknown", "Error output TransactionException::StackUnderflow");
	buffer.str(std::string());
}

BOOST_AUTO_TEST_CASE(toTransactionExceptionConvert)
{
	RLPException rlpEx("exception");//toTransactionException(*(dynamic_cast<Exception*>
	BOOST_CHECK_MESSAGE(toTransactionException(rlpEx) == TransactionException::BadRLP, "RLPException !=> TransactionException");
	OutOfGasIntrinsic oogEx;
	BOOST_CHECK_MESSAGE(toTransactionException(oogEx) == TransactionException::OutOfGasIntrinsic, "OutOfGasIntrinsic !=> TransactionException");
	InvalidSignature sigEx;
	BOOST_CHECK_MESSAGE(toTransactionException(sigEx) == TransactionException::InvalidSignature, "InvalidSignature !=> TransactionException");
	OutOfGasBase oogbEx;
	BOOST_CHECK_MESSAGE(toTransactionException(oogbEx) == TransactionException::OutOfGasBase, "OutOfGasBase !=> TransactionException");
	InvalidNonce nonceEx;
	BOOST_CHECK_MESSAGE(toTransactionException(nonceEx) == TransactionException::InvalidNonce, "InvalidNonce !=> TransactionException");
	NotEnoughCash cashEx;
	BOOST_CHECK_MESSAGE(toTransactionException(cashEx) == TransactionException::NotEnoughCash, "NotEnoughCash !=> TransactionException");
	BlockGasLimitReached blGasEx;
	BOOST_CHECK_MESSAGE(toTransactionException(blGasEx) == TransactionException::BlockGasLimitReached, "BlockGasLimitReached !=> TransactionException");
	BadInstruction badInsEx;
	BOOST_CHECK_MESSAGE(toTransactionException(badInsEx) == TransactionException::BadInstruction, "BadInstruction !=> TransactionException");
	BadJumpDestination badJumpEx;
	BOOST_CHECK_MESSAGE(toTransactionException(badJumpEx) == TransactionException::BadJumpDestination, "BadJumpDestination !=> TransactionException");
	OutOfGas oogEx2;
	BOOST_CHECK_MESSAGE(toTransactionException(oogEx2) == TransactionException::OutOfGas, "OutOfGas !=> TransactionException");
	OutOfStack oosEx;
	BOOST_CHECK_MESSAGE(toTransactionException(oosEx) == TransactionException::OutOfStack, "OutOfStack !=> TransactionException");
	StackUnderflow stackEx;
	BOOST_CHECK_MESSAGE(toTransactionException(stackEx) == TransactionException::StackUnderflow, "StackUnderflow !=> TransactionException");
	Exception notEx;
	BOOST_CHECK_MESSAGE(toTransactionException(notEx) == TransactionException::Unknown, "Unexpected should be TransactionException::Unknown");
}

BOOST_AUTO_TEST_SUITE_END()
