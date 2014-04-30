#pragma once

#include <libethcore/Exceptions.h>

namespace eth
{

class VMException: public Exception {};
class StepsDone: public VMException {};
class BreakPointHit: public VMException {};
class BadInstruction: public VMException {};
class OutOfGas: public VMException {};
class StackTooSmall: public VMException { public: StackTooSmall(u256 _req, u256 _got): req(_req), got(_got) {} u256 req; u256 got; };
class OperandOutOfRange: public VMException { public: OperandOutOfRange(u256 _min, u256 _max, u256 _got): mn(_min), mx(_max), got(_got) {} u256 mn; u256 mx; u256 got; };

class NotEnoughCash: public Exception {};

class GasPriceTooLow: public Exception {};
class BlockGasLimitReached: public Exception {};
class NoSuchContract: public Exception {};
class ContractAddressCollision: public Exception {};
class FeeTooSmall: public Exception {};
class TooMuchGasUsed: public Exception {};
class ExtraDataTooBig: public Exception {};
class InvalidSignature: public Exception {};
class InvalidTransactionFormat: public Exception { public: InvalidTransactionFormat(int _f, bytesConstRef _d): m_f(_f), m_d(_d.toBytes()) {} int m_f; bytes m_d; virtual std::string description() const { return "Invalid transaction format: Bad field " + toString(m_f) + " (" + toHex(m_d) + ")"; } };
class InvalidBlockFormat: public Exception { public: InvalidBlockFormat(int _f, bytesConstRef _d): m_f(_f), m_d(_d.toBytes()) {} int m_f; bytes m_d; virtual std::string description() const { return "Invalid block format: Bad field " + toString(m_f) + " (" + toHex(m_d) + ")"; } };
class InvalidBlockHeaderFormat: public Exception { public: InvalidBlockHeaderFormat(int _f, bytesConstRef _d): m_f(_f), m_d(_d.toBytes()) {} int m_f; bytes m_d; virtual std::string description() const { return "Invalid block header format: Bad field " + toString(m_f) + " (" + toHex(m_d) + ")"; } };
class InvalidUnclesHash: public Exception {};
class InvalidUncle: public Exception {};
class UncleNotAnUncle: public Exception {};
class DuplicateUncleNonce: public Exception {};
class InvalidStateRoot: public Exception {};
class InvalidTransactionsHash: public Exception { public: InvalidTransactionsHash(h256 _head, h256 _real): m_head(_head), m_real(_real) {} h256 m_head; h256 m_real; virtual std::string description() const { return "Invalid transactions hash:  header says: " + toHex(m_head.ref()) + " block is:" + toHex(m_real.ref()); } };
class InvalidTransaction: public Exception {};
class InvalidDifficulty: public Exception {};
class InvalidGasLimit: public Exception {};
class InvalidMinGasPrice: public Exception { public: InvalidMinGasPrice(u256 _provided = 0, u256 _limit = 0): provided(_provided), limit(_limit) {} u256 provided; u256 limit; virtual std::string description() const { return "Invalid minimum gas price (provided: " + toString(provided) + " limit:" + toString(limit) + ")"; } };
class InvalidTransactionGasUsed: public Exception {};
class InvalidTransactionStateRoot: public Exception {};
class InvalidTimestamp: public Exception {};
class InvalidNonce: public Exception { public: InvalidNonce(u256 _required = 0, u256 _candidate = 0): required(_required), candidate(_candidate) {} u256 required; u256 candidate; virtual std::string description() const { return "Invalid nonce (r: " + toString(required) + " c:" + toString(candidate) + ")"; } };
class InvalidBlockNonce: public Exception { public: InvalidBlockNonce(h256 _h = h256(), h256 _n = h256(), u256 _d = 0): h(_h), n(_n), d(_d) {} h256 h; h256 n; u256 d; virtual std::string description() const { return "Invalid nonce (h: " + toString(h) + " n:" + toString(n) + " d:" + toString(d) + ")"; } };
class InvalidParentHash: public Exception {};
class InvalidNumber: public Exception {};
class InvalidContractAddress: public Exception {};

}
