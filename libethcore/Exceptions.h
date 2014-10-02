#pragma once

#include <libdevcore/Exceptions.h>

namespace dev
{
namespace eth
{

// information to add to exceptions
typedef boost::error_info<struct tag_field, std::string> errinfo_name;
typedef boost::error_info<struct tag_field, int> errinfo_field;
typedef boost::error_info<struct tag_data, std::string> errinfo_data;
typedef boost::tuple<errinfo_field, errinfo_data> BadFieldError;

struct DatabaseAlreadyOpen: virtual dev::Exception {};
struct NotEnoughCash: virtual dev::Exception {};
struct GasPriceTooLow: virtual dev::Exception {};
struct BlockGasLimitReached: virtual dev::Exception {};
struct NoSuchContract: virtual dev::Exception {};
struct ContractAddressCollision: virtual dev::Exception {};
struct FeeTooSmall: virtual dev::Exception {};
struct TooMuchGasUsed: virtual dev::Exception {};
struct ExtraDataTooBig: virtual dev::Exception {};
struct InvalidSignature: virtual dev::Exception {};
class InvalidBlockFormat: public dev::Exception { public: InvalidBlockFormat(int _f, bytesConstRef _d): m_f(_f), m_d(_d.toBytes()) {} int m_f; bytes m_d; virtual const char* what() const noexcept { return ("Invalid block format: Bad field " + toString(m_f) + " (" + toHex(m_d) + ")").c_str(); } };
struct InvalidUnclesHash: virtual dev::Exception {};
struct InvalidUncle: virtual dev::Exception {};
struct UncleTooOld: virtual dev::Exception {};
class UncleInChain: public dev::Exception { public: UncleInChain(h256Set _uncles, h256 _block): m_uncles(_uncles), m_block(_block) {} h256Set m_uncles; h256 m_block;virtual const char* what() const noexcept { return ("Uncle in block already mentioned: Uncles " + toString(m_uncles) + " (" + m_block.abridged() + ")").c_str(); } };
struct DuplicateUncleNonce: virtual dev::Exception {};
struct InvalidStateRoot: virtual dev::Exception {};
class InvalidTransactionsHash: public dev::Exception { public: InvalidTransactionsHash(h256 _head, h256 _real): m_head(_head), m_real(_real) {} h256 m_head; h256 m_real; virtual const char* what() const noexcept { return ("Invalid transactions hash:  header says: " + toHex(m_head.ref()) + " block is:" + toHex(m_real.ref())).c_str(); } };
struct InvalidTransaction: virtual dev::Exception {};
struct InvalidDifficulty: virtual dev::Exception {};
class InvalidGasLimit: public dev::Exception { public: InvalidGasLimit(u256 _provided = 0, u256 _valid = 0): provided(_provided), valid(_valid) {} u256 provided; u256 valid; virtual const char* what() const noexcept { return ("Invalid gas limit (provided: " + toString(provided) + " valid:" + toString(valid) + ")").c_str(); } };
class InvalidMinGasPrice: public dev::Exception { public: InvalidMinGasPrice(u256 _provided = 0, u256 _limit = 0): provided(_provided), limit(_limit) {} u256 provided; u256 limit; virtual const char* what() const noexcept { return ("Invalid minimum gas price (provided: " + toString(provided) + " limit:" + toString(limit) + ")").c_str(); } };
struct InvalidTransactionGasUsed: virtual dev::Exception {};
struct InvalidTransactionStateRoot: virtual dev::Exception {};
struct InvalidTimestamp: virtual dev::Exception {};
class InvalidNonce: public dev::Exception { public: InvalidNonce(u256 _required = 0, u256 _candidate = 0): required(_required), candidate(_candidate) {} u256 required; u256 candidate; virtual const char* what() const noexcept { return ("Invalid nonce (r: " + toString(required) + " c:" + toString(candidate) + ")").c_str(); } };
class InvalidBlockNonce: public dev::Exception { public: InvalidBlockNonce(h256 _h = h256(), h256 _n = h256(), u256 _d = 0): h(_h), n(_n), d(_d) {} h256 h; h256 n; u256 d; virtual const char* what() const noexcept { return ("Invalid nonce (h: " + toString(h) + " n:" + toString(n) + " d:" + toString(d) + ")").c_str(); } };
struct InvalidParentHash: virtual dev::Exception {};
struct InvalidNumber: virtual dev::Exception {};
struct InvalidContractAddress: virtual dev::Exception {};

}
}
