#include <QtQml/QtQml>
#include <QtCore/QtCore>
#include <libethcore/FileSystem.h>
#include <libethereum/Dagger.h>
#include <libethereum/Client.h>
#include <libethereum/Instruction.h>
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
using eth::assemble;
using eth::compileLisp;
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
	return client()->postState().isContractAddress(_a);
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

void QmlEthereum::transact(Secret _secret, u256 _amount, u256 _gasPrice, u256 _gas, QByteArray _code, QByteArray _init)
{
	client()->transact(_secret, _amount, bytes(_code.data(), _code.data() + _code.size()), bytes(_init.data(), _init.data() + _init.size()), _gas, _gasPrice);
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


QEthereum::QEthereum(QObject* _p, Client* _c, QList<eth::KeyPair> _accounts): QObject(_p), m_client(_c), m_accounts(_accounts)
{
	connect(_p, SIGNAL(changed()), SIGNAL(changed()));
}

QEthereum::~QEthereum()
{
}

Client* QEthereum::client() const
{
	return m_client;
}

QVariant QEthereum::coinbase() const
{
	return toQJS(client()->address());
}

QVariant QEthereum::account() const
{
	if (m_accounts.empty())
		return toQJS(Address());
	return toQJS(m_accounts[0].address());
}

QList<QVariant> QEthereum::accounts() const
{
	QList<QVariant> ret;
	for (auto i: m_accounts)
		ret.push_back(toQJS(i.address()));
	return ret;
}

QVariant QEthereum::key() const
{
	if (m_accounts.empty())
		return toQJS(KeyPair());
	return toQJS(m_accounts[0]);
}

QList<QVariant> QEthereum::keys() const
{
	QList<QVariant> ret;
	for (auto i: m_accounts)
		ret.push_back(toQJS(i));
	return ret;
}

void QEthereum::setCoinbase(QVariant _a)
{
	if (client()->address() != to<Address>(_a))
	{
		client()->setAddress(to<Address>(_a));
		changed();
	}
}

QVariant QEthereum::balanceAt(QVariant _a) const
{
	return toQJS(client()->postState().balance(to<Address>(_a)));
}

QVariant QEthereum::storageAt(QVariant _a, QVariant _p) const
{
	return toQJS(client()->postState().contractStorage(to<Address>(_a), to<u256>(_p)));
}

u256 QEthereum::balanceAt(Address _a) const
{
	return client()->postState().balance(_a);
}

bool QEthereum::isContractAt(QVariant _a) const
{
	return client()->postState().isContractAddress(to<Address>(_a));
}

bool QEthereum::isContractAt(Address _a) const
{
	return client()->postState().isContractAddress(_a);
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

double QEthereum::txCountAt(QVariant _a) const
{
	return (double)client()->postState().transactionsFrom(to<Address>(_a));
}

double QEthereum::txCountAt(Address _a) const
{
	return (double)client()->postState().transactionsFrom(_a);
}

unsigned QEthereum::peerCount() const
{
	return (unsigned)client()->peerCount();
}

QVariant QEthereum::create(QVariant _secret, QVariant _amount, QByteArray _code, QByteArray _init, QVariant _gas, QVariant _gasPrice)
{
	return toQJS(client()->transact(to<Secret>(_secret), to<u256>(_amount), bytes(_code.data(), _code.data() + _code.size()), bytes(_init.data(), _init.data() + _init.size()), to<u256>(_gas), to<u256>(_gasPrice)));
}

void QEthereum::transact(QVariant _secret, QVariant _amount, QVariant _dest, QByteArray _data, QVariant _gas, QVariant _gasPrice)
{
	client()->transact(to<Secret>(_secret), to<u256>(_amount), to<Address>(_dest), bytes(_data.data(), _data.data() + _data.size()), to<u256>(_gas), to<u256>(_gasPrice));
}

// extra bits needed to link on VS
#ifdef _MSC_VER

// include moc file, ofuscated to hide from automoc
#include\
"moc_QEthereum.cpp"

// specify library dependencies, it's easier to do here than in the project since we can control the "d" debug suffix
#ifdef _DEBUG
#define QTLIB(x) x"d.lib"
#else 
#define QTLIB(x) x".lib"
#endif

#pragma comment(lib, QTLIB("Qt5PlatformSupport"))
#pragma comment(lib, QTLIB("Qt5Core"))
#pragma comment(lib, QTLIB("Qt5GUI"))
#pragma comment(lib, QTLIB("Qt5Widgets"))
#pragma comment(lib, QTLIB("Qt5Network"))

#endif
