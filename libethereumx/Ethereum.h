/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file Ethereum.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <thread>
#include <mutex>
#include <list>
#include <atomic>
#include <boost/utility.hpp>
#include <libethential/Common.h>
#include <libethential/CommonIO.h>
#include <libethential/Guards.h>
#include <libevm/FeeStructure.h>
#include <libp2p/Common.h>
#include <libethcore/Dagger.h>
#include <libethereum/PastMessage.h>
#include <libethereum/MessageFilter.h>
#include <libethereum/CommonNet.h>

namespace dev
{
namespace eth
{

class Client;

/**
 * @brief Main API hub for interfacing with Ethereum.
 * This class is automatically able to share a single machine-wide Client instance with other
 * instances, cross-process.
 *
 * Other than that, it provides much the same subset of functionality as Client.
 */
class Ethereum
{
	friend class OldMiner;

public:
	/// Constructor. After this, everything should be set up to go.
	Ethereum();

	/// Destructor.
	~Ethereum();

	/// Submits the given message-call transaction.
	void submitTransaction(Secret const& _secret, u256 _value, Address _dest, bytes const& _data = bytes(), u256 _gas = 10000, u256 _gasPrice = c_defaultGasPrice);

	/// Submits a new contract-creation transaction.
	/// @returns the new contract's address (assuming it all goes through).
	Address submitTransaction(Secret const& _secret, u256 _endowment, bytes const& _init, u256 _gas = 10000, u256 _gasPrice = c_defaultGasPrice);

	/// Injects the RLP-encoded transaction given by the _rlp into the transaction queue directly.
	void inject(bytesConstRef _rlp);

	/// Blocks until all pending transactions have been processed.
	void flushTransactions();

	/// Makes the given call. Nothing is recorded into the state.
	bytes call(Address const& _from, u256 _value, Address _dest, bytes const& _data = bytes(), u256 _gas = 10000, u256 _gasPrice = c_defaultGasPrice);

	// Informational stuff

	// [NEW API]

	int getDefault() const { return m_default; }
	void setDefault(int _block) { m_default = _block; }

	u256 balanceAt(Address _a) const { return balanceAt(_a, m_default); }
	u256 countAt(Address _a) const { return countAt(_a, m_default); }
	u256 stateAt(Address _a, u256 _l) const { return stateAt(_a, _l, m_default); }
	bytes codeAt(Address _a) const { return codeAt(_a, m_default); }
	std::map<u256, u256> storageAt(Address _a) const { return storageAt(_a, m_default); }

	u256 balanceAt(Address _a, int _block) const;
	u256 countAt(Address _a, int _block) const;
	u256 stateAt(Address _a, u256 _l, int _block) const;
	bytes codeAt(Address _a, int _block) const;
	std::map<u256, u256> storageAt(Address _a, int _block) const;

	PastMessages messages(MessageFilter const& _filter) const;

	// [EXTRA API]:
#if 0
	/// Get a map containing each of the pending transactions.
	/// @TODO: Remove in favour of transactions().
	Transactions pending() const { return m_postMine.pending(); }

	/// Differences between transactions.
	StateDiff diff(unsigned _txi) const { return diff(_txi, m_default); }
	StateDiff diff(unsigned _txi, h256 _block) const;
	StateDiff diff(unsigned _txi, int _block) const;

	/// Get a list of all active addresses.
	std::vector<Address> addresses() const { return addresses(m_default); }
	std::vector<Address> addresses(int _block) const;

	/// Get the fee associated for a transaction with the given data.
	static u256 txGas(uint _dataCount, u256 _gas = 0) { return c_txDataGas * _dataCount + c_txGas + _gas; }

	/// Get the remaining gas limit in this block.
	u256 gasLimitRemaining() const { return m_postMine.gasLimitRemaining(); }
#endif
	// Network stuff:

	/// Get information on the current peer set.
	std::vector<p2p::PeerInfo> peers();
	/// Same as peers().size(), but more efficient.
	size_t peerCount() const;

	/// Connect to a particular peer.
	void connect(std::string const& _seedHost, unsigned short _port = 30303);

private:
	/// Ensure that through either the client or the
	void ensureReady();
	/// Check to see if the client/server connection is open.
	bool connectionOpen() const;
	/// Start the API client.
	void connectToRPCServer();
	/// Start the API server.
	void startRPCServer();

	std::unique_ptr<Client> m_client;

	int m_default = -1;
};

}
}
