#pragma once

#include <exception>
#include "Common.h"

namespace eth
{

class BadHexCharacter: public std::exception {};
class NotEnoughCash: public std::exception {};
class BadInstruction: public std::exception {};
class StackTooSmall: public std::exception { public: StackTooSmall(u256 _req, u256 _got): req(_req), got(_got) {} u256 req; u256 got; };
class OperandOutOfRange: public std::exception { public: OperandOutOfRange(u256 _min, u256 _max, u256 _got): mn(_min), mx(_max), got(_got) {} u256 mn; u256 mx; u256 got; };
class ExecutionException: public std::exception {};
class NoSuchContract: public std::exception {};
class ContractAddressCollision: public std::exception {};
class FeeTooSmall: public std::exception {};
class InvalidSignature: public std::exception {};
class InvalidTransactionFormat: public std::exception {};
class InvalidBlockFormat: public std::exception {};
class InvalidUnclesHash: public std::exception {};
class InvalidTransactionsHash: public std::exception {};
class InvalidTransaction: public std::exception {};
class InvalidDifficulty: public std::exception {};
class InvalidTimestamp: public std::exception {};
class InvalidNonce: public std::exception {};

}
