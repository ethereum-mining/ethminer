#pragma once

#include <libethential/Exceptions.h>

namespace dev
{
namespace eth
{

class DatabaseAlreadyOpen: public dev::Exception {};

class NotEnoughCash: public dev::Exception {};

class GasPriceTooLow: public dev::Exception {};
class BlockGasLimitReached: public dev::Exception {};
class NoSuchContract: public dev::Exception {};
class ContractAddressCollision: public dev::Exception {};
class FeeTooSmall: public dev::Exception {};
class TooMuchGasUsed: public dev::Exception {};
class ExtraDataTooBig: public dev::Exception {};
class InvalidSignature: public dev::Exception {};
class InvalidTransactionFormat: public dev::Exception { public: InvalidTransactionFormat(int _f, bytesConstRef _d): m_f(_f), m_d(_d.toBytes()) {} int m_f; bytes m_d; virtual std::string description() const { return "Invalid transaction format: Bad field " + toString(m_f) + " (" + toHex(m_d) + ")"; } };
class InvalidBlockFormat: public dev::Exception { public: InvalidBlockFormat(int _f, bytesConstRef _d): m_f(_f), m_d(_d.toBytes()) {} int m_f; bytes m_d; virtual std::string description() const { return "Invalid block format: Bad field " + toString(m_f) + " (" + toHex(m_d) + ")"; } };
class InvalidBlockHeaderFormat: public dev::Exception { public: InvalidBlockHeaderFormat(int _f, bytesConstRef _d): m_f(_f), m_d(_d.toBytes()) {} int m_f; bytes m_d; virtual std::string description() const { return "Invalid block header format: Bad field " + toString(m_f) + " (" + toHex(m_d) + ")"; } };
class InvalidUnclesHash: public dev::Exception {};
class InvalidUncle: public dev::Exception {};
class UncleTooOld: public dev::Exception {};
class UncleInChain: public dev::Exception {};
class DuplicateUncleNonce: public dev::Exception {};
class InvalidStateRoot: public dev::Exception {};
class InvalidTransactionsHash: public dev::Exception { public: InvalidTransactionsHash(h256 _head, h256 _real): m_head(_head), m_real(_real) {} h256 m_head; h256 m_real; virtual std::string description() const { return "Invalid transactions hash:  header says: " + toHex(m_head.ref()) + " block is:" + toHex(m_real.ref()); } };
class InvalidTransaction: public dev::Exception {};
class InvalidDifficulty: public dev::Exception {};
class InvalidGasLimit: public dev::Exception { public: InvalidGasLimit(u256 _provided = 0, u256 _valid = 0): provided(_provided), valid(_valid) {} u256 provided; u256 valid; virtual std::string description() const { return "Invalid gas limit (provided: " + toString(provided) + " valid:" + toString(valid) + ")"; } };
class InvalidMinGasPrice: public dev::Exception { public: InvalidMinGasPrice(u256 _provided = 0, u256 _limit = 0): provided(_provided), limit(_limit) {} u256 provided; u256 limit; virtual std::string description() const { return "Invalid minimum gas price (provided: " + toString(provided) + " limit:" + toString(limit) + ")"; } };
class InvalidTransactionGasUsed: public dev::Exception {};
class InvalidTransactionStateRoot: public dev::Exception {};
class InvalidTimestamp: public dev::Exception {};
class InvalidNonce: public dev::Exception { public: InvalidNonce(u256 _required = 0, u256 _candidate = 0): required(_required), candidate(_candidate) {} u256 required; u256 candidate; virtual std::string description() const { return "Invalid nonce (r: " + toString(required) + " c:" + toString(candidate) + ")"; } };
class InvalidBlockNonce: public dev::Exception { public: InvalidBlockNonce(h256 _h = h256(), h256 _n = h256(), u256 _d = 0): h(_h), n(_n), d(_d) {} h256 h; h256 n; u256 d; virtual std::string description() const { return "Invalid nonce (h: " + toString(h) + " n:" + toString(n) + " d:" + toString(d) + ")"; } };
class InvalidParentHash: public dev::Exception {};
class InvalidNumber: public dev::Exception {};
class InvalidContractAddress: public dev::Exception {};

}
}
