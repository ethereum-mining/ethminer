#pragma once

#include <string>
#include <vector>
#include <tuple>
#include <libethereum/Interface.h>
#include "Common.h"
#include "CommonData.h"

namespace dev {
namespace eth {

template <unsigned S> std::string toJS(FixedHash<S> const& _h) { return "0x" + toHex(_h.ref()); }
template <unsigned N> std::string toJS(boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N, N, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>> const& _n) { return "0x" + toHex(toCompactBigEndian(_n)); }
inline std::string toJS(dev::bytes const& _n) { return "0x" + dev::toHex(_n); }

bytes jsToBytes(std::string const& _s);
std::string jsPadded(std::string const& _s, unsigned _l, unsigned _r);
std::string jsPadded(std::string const& _s, unsigned _l);
std::string jsUnpadded(std::string _s);

template <unsigned N> FixedHash<N> jsToFixed(std::string const& _s)
{
    if (_s.substr(0, 2) == "0x")
        // Hex
        return FixedHash<N>(_s.substr(2));
    else if (_s.find_first_not_of("0123456789") == std::string::npos)
        // Decimal
        return (typename FixedHash<N>::Arith)(_s);
    else
        // Binary
        return FixedHash<N>(asBytes(jsPadded(_s, N)));
}

inline std::string jsToFixed(double _s)
{
    return toJS(dev::u256(_s * (double)(dev::u256(1) << 128)));
}


template <unsigned N> boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>> jsToInt(std::string const& _s)
{
    if (_s.substr(0, 2) == "0x")
        // Hex
        return fromBigEndian<boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>>(fromHex(_s.substr(2)));
    else if (_s.find_first_not_of("0123456789") == std::string::npos)
        // Decimal
        return boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>(_s);
    else
        // Binary
        return fromBigEndian<boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>>(asBytes(jsPadded(_s, N)));
}

inline Address jsToAddress(std::string const& _s) { return jsToFixed<20>(_s); }
inline Secret jsToSecret(std::string const& _s) { return jsToFixed<32>(_s); }
inline u256 jsToU256(std::string const& _s) { return jsToInt<32>(_s); }

inline std::string jsFromBinary(dev::bytes _s, unsigned _padding = 32)
{
    _s.resize(std::max<unsigned>(_s.size(), _padding));
    return "0x" + dev::toHex(_s);
}

inline std::string jsFromBinary(std::string const& _s, unsigned _padding = 32)
{
    return jsFromBinary(asBytes(_s), _padding);
}

// we note really need KeyPair from
// but it usefull for checking the balance
struct TransactionSkeleton
{
    dev::KeyPair from;
    Address to;
    u256 value;
    bytes data;
    u256 gas;
    u256 gasPrice;
};


class CommonJS
{
public:
    CommonJS(dev::eth::Interface* _c) : m_client(_c) {}
    dev::eth::Interface* client() const;
    void setAccounts(std::vector<dev::KeyPair> _accounts);

    std::string ethTest() const;

    // properties
    std::string coinbase() const;
    bool isListening() const;
    void setListening(bool _l);
    bool isMining() const;
    void setMining(bool _l);
    std::string /*dev::u256*/ gasPrice() const;
    std::string /*dev::KeyPair*/ key() const;
    std::vector<std::string> /*list of dev::KeyPair*/ keys() const;
    unsigned peerCount() const;
    int defaultBlock() const;
    unsigned /*dev::u256*/ number() const;

    // synchronous getters
    std::string balanceAt(std::string const &_a, int _block) const;
    std::string stateAt(std::string const &_a, std::string &_p, int _block) const;
    double countAt(std::string const &_a, int _block) const;
    std::string codeAt(std::string const &_a, int _block) const;

    // transactions
    std::string transact(dev::eth::TransactionSkeleton _t);
    std::string call(dev::eth::TransactionSkeleton _t);

    // blockchain
    std::tuple<BlockInfo, BlockDetails> block(std::string const &_numberOrHash) const;
    std::string transaction(std::string const &_numberOrHash, int _index) const;
    std::string uncle(std::string const &_numberOrHash, int _index) const;

    // watches and messages filtering
    std::string messages(std::string const &_attribs) const;
    // TODO watch

    // misc
    std::string secretToAddress(std::string const &_a) const;
    std::string lll(std::string const &_l) const;
    std::string sha3(std::string const &_s) const;



private:
    dev::eth::Interface* m_client;
    std::vector<dev::KeyPair> m_accounts;
};
}
}

