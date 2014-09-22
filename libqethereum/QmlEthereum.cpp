#if ETH_QTQML
#include <QtQml/QtQml>
#endif
#include <QtCore/QtCore>
#include <QtWebKitWidgets/QWebFrame>
#include <libdevcrypto/FileSystem.h>
#include <libethcore/Dagger.h>
#include <libevmface/Instruction.h>
#include <liblll/Compiler.h>
#include <libethereum/Client.h>
#include <libethereum/EthereumHost.h>
#include "QmlEthereum.h"
using namespace std;

// types
using dev::bytes;
using dev::bytesConstRef;
using dev::h160;
using dev::h256;
using dev::u160;
using dev::u256;
using dev::u256s;
using dev::RLP;
using dev::Address;
using dev::eth::BlockInfo;
using dev::eth::Client;
using dev::eth::Instruction;
using dev::KeyPair;
using dev::eth::NodeMode;
using dev::Secret;
using dev::eth::Transaction;
using dev::p2p::PeerInfo;

// functions
using dev::toHex;
using dev::fromHex;
using dev::right160;
using dev::simpleDebugOut;
using dev::toLog2;
using dev::toString;
using dev::eth::disassemble;
using dev::eth::formatBalance;
using dev::eth::units;

// vars
using dev::g_logPost;
using dev::g_logVerbosity;

// Horrible global for the mainwindow. Needed for the QmlEthereums to find the Main window which acts as multiplexer for now.
// Can get rid of this once we've sorted out ITC for signalling & multiplexed querying.
dev::eth::Client* g_qmlClient;
QObject* g_qmlMain;
#if 0
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
//	if (m_eth)
//		disconnect(m_eth, SIGNAL(changed()), this, SIGNAL(changed()));
	m_eth = _eth;
//	if (m_eth)
//		connect(m_eth, SIGNAL(changed()), this, SIGNAL(changed()));
	ethChanged();
//	changed();
}

dev::u256 QmlAccount::balance() const
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
//	connect(g_qmlMain, SIGNAL(changed()), SIGNAL(changed()));
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
//		changed();
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

#endif

// extra bits needed to link on VS
#ifdef _MSC_VER

// include moc file, ofuscated to hide from automoc
#include\
"moc_QmlEthereum.cpp"

#endif
