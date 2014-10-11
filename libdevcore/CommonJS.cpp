#include "CommonJS.h"

namespace dev {
namespace eth {

bytes dev::eth::jsToBytes(std::string const& _s)
{
    if (_s.substr(0, 2) == "0x")
        // Hex
        return fromHex(_s.substr(2));
    else if (_s.find_first_not_of("0123456789") == std::string::npos)
        // Decimal
        return toCompactBigEndian(bigint(_s));
    else
        // Binary
        return asBytes(_s);
}

std::string dev::eth::jsPadded(std::string const& _s, unsigned _l, unsigned _r)
{
    bytes b = jsToBytes(_s);
    while (b.size() < _l)
        b.insert(b.begin(), 0);
    while (b.size() < _r)
        b.push_back(0);
    return asString(b).substr(b.size() - std::max(_l, _r));
}

std::string dev::eth::jsPadded(std::string const& _s, unsigned _l)
{
    if (_s.substr(0, 2) == "0x" || _s.find_first_not_of("0123456789") == std::string::npos)
        // Numeric: pad to right
        return jsPadded(_s, _l, _l);
    else
        // Text: pad to the left
        return jsPadded(_s, 0, _l);
}

std::string dev::eth::jsUnpadded(std::string _s)
{
    auto p = _s.find_last_not_of((char)0);
    _s.resize(p == std::string::npos ? 0 : (p + 1));
    return _s;
}

dev::eth::Interface* CommonJS::client() const
{
    return m_client;
}

void CommonJS::setAccounts(std::vector<dev::KeyPair> _accounts)
{
    m_accounts = _accounts;
}

std::string CommonJS::ethTest() const
{
    return "Hello World!";
}

std::string CommonJS::coinbase() const
{
    return m_client ? toJS(client()->address()) : "";
}

bool CommonJS::isListening() const
{
    return /*m_client ? client()->haveNetwork() :*/ false;
}

void CommonJS::setListening(bool _l)
{
    if (!m_client)
        return;
/*	if (_l)
        client()->startNetwork();
    else
        client()->stopNetwork();*/
}

bool CommonJS::isMining() const
{
    return m_client ? client()->isMining() : false;
}

void CommonJS::setMining(bool _l)
{
    if (m_client)
    {
        if (_l)
            client()->startMining();
        else
            client()->stopMining();
    }
}

std::string CommonJS::gasPrice() const
{
    return toJS(10 * dev::eth::szabo);
}

std::string CommonJS::key() const
{
    if (m_accounts.empty())
        return toJS(KeyPair().sec());
    return toJS(m_accounts[0].sec());
}

std::vector<std::string> CommonJS::keys() const
{
    std::vector<std::string> ret;
    for (auto i: m_accounts)
        ret.push_back(toJS(i.sec()));
    return ret;
}

unsigned CommonJS::peerCount() const
{
    return /*m_client ? (unsigned)client()->peerCount() :*/ 0;
}

int CommonJS::defaultBlock() const
{
    return m_client ? m_client->getDefault() : 0;
}

unsigned CommonJS::number() const
{
    return m_client ? client()->number() + 1 : 0;
}

std::string CommonJS::balanceAt(const std::string &_a, int _block) const
{
    return m_client ? toJS(client()->balanceAt(jsToAddress(_a), _block)) : "";
}

std::string CommonJS::stateAt(const std::string &_a, std::string &_p, int _block) const
{
    return m_client ? toJS(client()->stateAt(jsToAddress(_a), jsToU256(_p), _block)) : "";
}

double CommonJS::countAt(const std::string &_a, int _block) const
{
    return m_client ? (double)(uint64_t)client()->countAt(jsToAddress(_a), _block) : 0;
}

std::string CommonJS::codeAt(const std::string &_a, int _block) const
{
    return m_client ? jsFromBinary(client()->codeAt(jsToAddress(_a), _block)) : "";
}

std::string CommonJS::transact(TransactionSkeleton _t)
{
    std::string ret;
    if (!m_client)
        return ret;
    // what if there is no from accout specified?
    if (!_t.from.secret() && m_accounts.size())
    {
        auto b = m_accounts.front();
        for (auto a: m_accounts)
            if (client()->balanceAt(KeyPair(a).address()) > client()->balanceAt(KeyPair(b).address()))
                b = a;
        _t.from = b;
    }
    if (!_t.gasPrice)
        _t.gasPrice = 10 * dev::eth::szabo;
    if (!_t.gas)
        _t.gas = std::min<u256>(client()->gasLimitRemaining(), client()->balanceAt(KeyPair(_t.from).address()) / _t.gasPrice);
    if (_t.to)
        client()->transact(_t.from.secret(), _t.value, _t.to, _t.data, _t.gas, _t.gasPrice);
    else
        ret = toJS(client()->transact(_t.from.secret(), _t.value, _t.data, _t.gas, _t.gasPrice));
    client()->flushTransactions();
    return ret;
}

std::string CommonJS::call(TransactionSkeleton _t)
{
    std::string ret;
    if (!m_client)
        return ret;
    if (!_t.to)
        return ret;
    if (!_t.from.secret() && m_accounts.size())
        _t.from = m_accounts[0];
    if (!_t.gasPrice)
        _t.gasPrice = 10 * dev::eth::szabo;
    if (!_t.gas)
        _t.gas = client()->balanceAt(_t.from.address()) / _t.gasPrice;
    ret = toJS(client()->call(_t.from.secret(), _t.value, _t.to, _t.data, _t.gas, _t.gasPrice));
    return ret;
}

std::tuple<BlockInfo, BlockDetails> CommonJS::block(const std::string &_numberOrHash) const
{
    auto n = jsToU256(_numberOrHash);
    auto h = n < m_client->number() ? m_client->hashFromNumber((unsigned)n) : jsToFixed<32>(_numberOrHash);
    return std::make_tuple(m_client->blockInfo(h), m_client->blockDetails(h));
}




}
}
