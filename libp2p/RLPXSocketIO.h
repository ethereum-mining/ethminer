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
/** @file RLPXSocketIO.h
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2015
 */

#pragma once

#include "RLPXFrameWriter.h"
namespace ba = boost::asio;
namespace bi = boost::asio::ip;

namespace dev
{
namespace p2p
{

class RLPXSocketIO: public std::enable_shared_from_this<RLPXSocketIO>
{
public:
	static uint32_t const MinFrameSize;
	static uint32_t const MaxPacketSize;
	static uint16_t const DefaultInitialCapacity;
	
	RLPXSocketIO(unsigned _protCount, RLPXFrameCoder& _coder, bi::tcp::socket& _socket, bool _flowControl = true, size_t _initialCapacity = DefaultInitialCapacity);

	void send(unsigned _protocolType, unsigned _type, RLPStream& _payload);
	
	void doWrite();
	
	bool congested() const { return !!m_congestion; }
	
private:
	static std::vector<RLPXFrameWriter> writers(unsigned _capacity);
	
	void deferWrite();
	
	void write(size_t _dequed);
	
	bool const m_flowControl;		///< True if flow control is enabled.
	
	RLPXFrameCoder& m_coder;		///< Encoder/decoder of frame payloads.
	bi::tcp::socket& m_socket;

	std::vector<bytes> m_toSend;	///< Reusable byte buffer for pending socket writes.
	
	std::vector<RLPXFrameWriter> m_writers;			///< Write queues for each protocol. TODO: map to bytes (of capability)
	std::unique_ptr<ba::deadline_timer> m_congestion;	///< Scheduled when writes are deferred due to congestion.

	Mutex x_queued;
	unsigned m_queued = 0;	///< Track total queued packets to ensure single write loop
	uint32_t m_egressCapacity;
};

}
}