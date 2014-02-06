#pragma once

#include <exception>
#include "Common.h"

namespace eth
{

class Exception: public std::exception
{
public:
	virtual std::string description() const { return "Unknown exception"; }
	virtual char const* what() const noexcept { return typeid(*this).name(); }
};

class BadHexCharacter: public Exception {};
class NotEnoughCash: public Exception {};
class BadInstruction: public Exception {};
class StackTooSmall: public Exception { public: StackTooSmall(u256 _req, u256 _got): req(_req), got(_got) {} u256 req; u256 got; };
class OperandOutOfRange: public Exception { public: OperandOutOfRange(u256 _min, u256 _max, u256 _got): mn(_min), mx(_max), got(_got) {} u256 mn; u256 mx; u256 got; };
class ExecutionException: public Exception {};
class NoSuchContract: public Exception {};
class ContractAddressCollision: public Exception {};
class FeeTooSmall: public Exception {};
class InvalidSignature: public Exception {};
class InvalidTransactionFormat: public Exception {};
class InvalidBlockFormat: public Exception { public: virtual std::string description() const { return "Invalid block format"; } };
class InvalidUnclesHash: public Exception {};
class InvalidUncle: public Exception {};
class InvalidStateRoot: public Exception {};
class InvalidTransactionsHash: public Exception {};
class InvalidTransaction: public Exception {};
class InvalidDifficulty: public Exception {};
class InvalidTimestamp: public Exception {};
class InvalidNonce: public Exception { public: InvalidNonce(u256 _required = 0, u256 _candidate = 0): required(_required), candidate(_candidate) {} u256 required; u256 candidate; };
class InvalidParentHash: public Exception {};
class InvalidContractAddress: public Exception {};
class NoNetworking: public Exception {};
class NoUPnPDevice: public Exception {};

}
