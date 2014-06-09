#include <QtQml/QtQml>
#include <QtCore/QtCore>
#include <QtWebKitWidgets/QWebFrame>
#include <libethsupport/FileSystem.h>
#include <libethcore/Dagger.h>
#include <libethcore/Instruction.h>
#include <liblll/Compiler.h>
#include <libethereum/Client.h>
#include <libethereum/PeerServer.h>
#include "QEthereum.h"
using namespace std;

// types
using eth::bytes;
using eth::bytesConstRef;
using eth::h160;
using eth::h256;
using eth::u160;
using eth::u256;
using eth::u256s;
using eth::Address;
using eth::BlockInfo;
using eth::Client;
using eth::Instruction;
using eth::KeyPair;
using eth::NodeMode;
using eth::PeerInfo;
using eth::RLP;
using eth::Secret;
using eth::Transaction;

// functions
using eth::toHex;
using eth::disassemble;
using eth::formatBalance;
using eth::fromHex;
using eth::right160;
using eth::simpleDebugOut;
using eth::toLog2;
using eth::toString;
using eth::units;

// vars
using eth::g_logPost;
using eth::g_logVerbosity;
using eth::c_instructionInfo;

// Horrible global for the mainwindow. Needed for the QmlEthereums to find the Main window which acts as multiplexer for now.
// Can get rid of this once we've sorted out ITC for signalling & multiplexed querying.
eth::Client* g_qmlClient;
QObject* g_qmlMain;

QmlAccount::QmlAccount(QObject*)
{
}

QmlAccount::~QmlAccount()
{
}

void QmlAccount::setEthereum(QmlEthereum* _eth)
{
	if (m_eth == _eth)
		return;
	if (m_eth)
		disconnect(m_eth, SIGNAL(changed()), this, SIGNAL(changed()));
	m_eth = _eth;
	if (m_eth)
		connect(m_eth, SIGNAL(changed()), this, SIGNAL(changed()));
	ethChanged();
	changed();
}

eth::u256 QmlAccount::balance() const
{
	if (m_eth)
		return m_eth->balanceAt(m_address);
	return u256(0);
}

double QmlAccount::txCount() const
{
	if (m_eth)
		return m_eth->txCountAt(m_address);
	return 0;
}

bool QmlAccount::isContract() const
{
	if (m_eth)
		return m_eth->isContractAt(m_address);
	return 0;
}

QmlEthereum::QmlEthereum(QObject* _p): QObject(_p)
{
	connect(g_qmlMain, SIGNAL(changed()), SIGNAL(changed()));
}

QmlEthereum::~QmlEthereum()
{
}

Client* QmlEthereum::client() const
{
	return g_qmlClient;
}

Address QmlEthereum::coinbase() const
{
	return client()->address();
}

void QmlEthereum::setCoinbase(Address _a)
{
	if (client()->address() != _a)
	{
		client()->setAddress(_a);
		changed();
	}
}

u256 QmlEthereum::balanceAt(Address _a) const
{
	return client()->postState().balance(_a);
}

bool QmlEthereum::isContractAt(Address _a) const
{
	return client()->postState().addressHasCode(_a);
}

bool QmlEthereum::isMining() const
{
	return client()->isMining();
}

bool QmlEthereum::isListening() const
{
	return client()->haveNetwork();
}

void QmlEthereum::setMining(bool _l)
{
	if (_l)
		client()->startMining();
	else
		client()->stopMining();
}

void QmlEthereum::setListening(bool _l)
{
	if (_l)
		client()->startNetwork();
	else
		client()->stopNetwork();
}

double QmlEthereum::txCountAt(Address _a) const
{
	return (double)client()->postState().transactionsFrom(_a);
}

unsigned QmlEthereum::peerCount() const
{
	return (unsigned)client()->peerCount();
}

void QmlEthereum::transact(Secret _secret, u256 _amount, u256 _gasPrice, u256 _gas, QByteArray _init)
{
	client()->transact(_secret, _amount, bytes(_init.data(), _init.data() + _init.size()), _gas, _gasPrice);
}

void QmlEthereum::transact(Secret _secret, Address _dest, u256 _amount, u256 _gasPrice, u256 _gas, QByteArray _data)
{
	client()->transact(_secret, _amount, _dest, bytes(_data.data(), _data.data() + _data.size()), _gas, _gasPrice);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

eth::bytes toBytes(QString const& _s)
{
	if (_s.startsWith("0x"))
		// Hex
		return eth::fromHex(_s.mid(2).toStdString());
	else if (!_s.contains(QRegExp("[^0-9]")))
		// Decimal
		return eth::toCompactBigEndian(eth::bigint(_s.toStdString()));
	else
		// Binary
		return asBytes(_s);
}

QString padded(QString const& _s, unsigned _l, unsigned _r)
{
	eth::bytes b = toBytes(_s);
	while (b.size() < _l)
		b.insert(b.begin(), 0);
	while (b.size() < _r)
		b.push_back(0);
	return asQString(eth::asBytes(eth::asString(b).substr(b.size() - max(_l, _r))));
}

//"0xff".bin().unbin()

QString QEthereum::secretToAddress(QString _s) const
{
	return toQJS(KeyPair(toSecret(_s)).address());
}

QString padded(QString const& _s, unsigned _l)
{
	if (_s.startsWith("0x") || !_s.contains(QRegExp("[^0-9]")))
		// Numeric: pad to right
		return padded(_s, _l, _l);
	else
		// Text: pad to the left
		return padded(_s, 0, _l);
}

QString unpadded(QString _s)
{
	while (_s.size() && _s.endsWith(QChar(0)))
		_s.chop(1);
	return _s;
}

QEthereum::QEthereum(QObject* _p, Client* _c, QList<eth::KeyPair> _accounts): QObject(_p), m_client(_c), m_accounts(_accounts)
{
	connect(_p, SIGNAL(changed()), SIGNAL(changed()));
}

QEthereum::~QEthereum()
{
}

void QEthereum::setup(QWebFrame* _e)
{
	// disconnect
	disconnect(SIGNAL(changed()));
	_e->addToJavaScriptWindowObject("eth", this, QWebFrame::ScriptOwnership);
/*	_e->addToJavaScriptWindowObject("u256", new U256Helper, QWebFrame::ScriptOwnership);
	_e->addToJavaScriptWindowObject("key", new KeyHelper, QWebFrame::ScriptOwnership);
	_e->addToJavaScriptWindowObject("bytes", new  BytesHelper, QWebFrame::ScriptOwnership);*/
	_e->evaluateJavaScript("eth.newBlock = function(f) { eth.changed.connect(f) }");
	_e->evaluateJavaScript("eth.watch = function(a, s, f) { eth.changed.connect(f ? f : s) }");
	_e->evaluateJavaScript("eth.create = function(s, v, c, g, p, f) { var v = eth.doCreate(s, v, c, g, p); if (f) f(v) }");
	_e->evaluateJavaScript("eth.transact = function(s, v, t, d, g, p, f) { eth.doTransact(s, v, t, d, g, p); if (f) f() }");
	_e->evaluateJavaScript("String.prototype.pad = function(l, r) { return eth.pad(this, l, r) }");
	_e->evaluateJavaScript("String.prototype.bin = function() { return eth.toBinary(this) }");
	_e->evaluateJavaScript("String.prototype.unbin = function(l) { return eth.fromBinary(this) }");
	_e->evaluateJavaScript("String.prototype.unpad = function(l) { return eth.unpad(this) }");
	_e->evaluateJavaScript("String.prototype.dec = function() { return eth.toDecimal(this) }");
	_e->evaluateJavaScript("String.prototype.sha3 = function() { return eth.sha3(this) }");
}

void QEthereum::teardown(QWebFrame*)
{
}

Client* QEthereum::client() const
{
	return m_client;
}

QString QEthereum::lll(QString _s) const
{
	return asQString(eth::compileLLL(_s.toStdString()));
}

QString QEthereum::sha3(QString _s) const
{
	return toQJS(eth::sha3(asBytes(_s)));
}

QString QEthereum::coinbase() const
{
	return toQJS(client()->address());
}

QString QEthereum::number() const
{
	return QString::number(client()->blockChain().number() + 1);
}

QString QEthereum::account() const
{
	if (m_accounts.empty())
		return toQJS(Address());
	return toQJS(m_accounts[0].address());
}

QStringList QEthereum::accounts() const
{
	QStringList ret;
	for (auto i: m_accounts)
		ret.push_back(toQJS(i.address()));
	return ret;
}

QString QEthereum::key() const
{
	if (m_accounts.empty())
		return toQJS(KeyPair().sec());
	return toQJS(m_accounts[0].sec());
}

QStringList QEthereum::keys() const
{
	QStringList ret;
	for (auto i: m_accounts)
		ret.push_back(toQJS(i.sec()));
	return ret;
}

void QEthereum::setCoinbase(QString _a)
{
	if (client()->address() != toAddress(_a))
	{
		client()->setAddress(toAddress(_a));
		changed();
	}
}

QString QEthereum::balanceAt(QString _a) const
{
	return toQJS(client()->postState().balance(toAddress(_a)));
}

QString QEthereum::storageAt(QString _a, QString _p) const
{
	return toQJS(client()->postState().storage(toAddress(_a), toU256(_p)));
}

u256 QEthereum::balanceAt(Address _a) const
{
	return client()->postState().balance(_a);
}

bool QEthereum::isContractAt(QString _a) const
{
	return client()->postState().addressHasCode(toAddress(_a));
}

bool QEthereum::isContractAt(Address _a) const
{
	return client()->postState().addressHasCode(_a);
}

bool QEthereum::isMining() const
{
	return client()->isMining();
}

bool QEthereum::isListening() const
{
	return client()->haveNetwork();
}

void QEthereum::setMining(bool _l)
{
	if (_l)
		client()->startMining();
	else
		client()->stopMining();
}

void QEthereum::setListening(bool _l)
{
	if (_l)
		client()->startNetwork();
	else
		client()->stopNetwork();
}

double QEthereum::txCountAt(QString _a) const
{
	return (double)client()->postState().transactionsFrom(toAddress(_a));
}

double QEthereum::txCountAt(Address _a) const
{
	return (double)client()->postState().transactionsFrom(_a);
}

unsigned QEthereum::peerCount() const
{
	return (unsigned)client()->peerCount();
}

QString QEthereum::doCreate(QString _secret, QString _amount, QString _init, QString _gas, QString _gasPrice)
{
	client()->changed();
	auto ret = toQJS(client()->transact(toSecret(_secret), toU256(_amount), toBytes(_init), toU256(_gas), toU256(_gasPrice)));
	while (!client()->peekChanged())
		this_thread::sleep_for(chrono::milliseconds(10));
	return ret;
}

void QEthereum::doTransact(QString _secret, QString _amount, QString _dest, QString _data, QString _gas, QString _gasPrice)
{
	client()->changed();
	client()->transact(toSecret(_secret), toU256(_amount), toAddress(_dest), toBytes(_data), toU256(_gas), toU256(_gasPrice));
	while (!client()->peekChanged())
		this_thread::sleep_for(chrono::milliseconds(10));
}

// extra bits needed to link on VS
#ifdef _MSC_VER

// include moc file, ofuscated to hide from automoc
#include\
"moc_QEthereum.cpp"

#endif
