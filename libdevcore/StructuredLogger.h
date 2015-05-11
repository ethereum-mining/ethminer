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
/** @file StructuredLogger.h
 * @author Lefteris Karapetsas <lefteris@ethdev.com>
 * @date 2015
 *
 * A simple helper class for the structured logging
 * The spec for the implemented log events is here:
 * https://github.com/ethereum/system-testing/wiki/Log-Events
 */

#pragma once

#include <fstream>
#include <string>
#include <chrono>

namespace Json { class Value; }
namespace boost { namespace asio { namespace ip { template<class T>class basic_endpoint; class tcp; }}}
namespace bi = boost::asio::ip;

namespace dev
{

// TODO: Make the output stream configurable. stdout, stderr, file e.t.c.
class StructuredLogger
{
public:
	/**
	 * Initializes the structured logger object
	 * @param _enabled        Whether logging is on or off
	 * @param _timeFormat     A time format string as described here:
	 *                        http://en.cppreference.com/w/cpp/chrono/c/strftime
	 *                        with which to display timestamps
	 */
	void initialize(bool _enabled, std::string const& _timeFormat, std::string const& _destinationURL = "");

	static StructuredLogger& get()
	{
		static StructuredLogger instance;
		return instance;
	}

	static void starting(std::string const& _clientImpl, const char* _ethVersion);
	static void stopping(std::string const& _clientImpl, const char* _ethVersion);
	static void p2pConnected(
		std::string const& _id,
		bi::basic_endpoint<bi::tcp> const& _addr,
		std::chrono::system_clock::time_point const& _ts,
		std::string const& _remoteVersion,
		unsigned int _numConnections
	);
	static void p2pDisconnected(
		std::string const& _id,
		bi::basic_endpoint<bi::tcp> const& _addr,
		unsigned int _numConnections
	);
	static void minedNewBlock(
		std::string const& _hash,
		std::string const& _blockNumber,
		std::string const& _chainHeadHash,
		std::string const& _prevHash
	);
	static void chainReceivedNewBlock(
		std::string const& _hash,
		std::string const& _blockNumber,
		std::string const& _chainHeadHash,
		std::string const& _remoteID,
		std::string const& _prevHash
	);
	static void chainNewHead(
		std::string const& _hash,
		std::string const& _blockNumber,
		std::string const& _chainHeadHash,
		std::string const& _prevHash
	);
	static void transactionReceived(std::string const& _hash, std::string const& _remoteId);
	// TODO: static void pendingQueueChanged(std::vector<h256> const& _hashes);
	// TODO: static void miningStarted();
	// TODO: static void stillMining(unsigned _hashrate);
	// TODO: static void miningStopped();

private:
	// Singleton class. Private default ctor and no copying
	StructuredLogger() = default;
	StructuredLogger(StructuredLogger const&) = delete;
	void operator=(StructuredLogger const&) = delete;

	void outputJson(Json::Value const& _value, std::string const& _name) const;

	bool m_enabled = false;
	std::string m_timeFormat = "%Y-%m-%dT%H:%M:%S";

	mutable std::ofstream m_out;
};

}
