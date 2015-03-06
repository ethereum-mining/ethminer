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
/** @file RLPXHandshake.h
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2015
 */


#pragma once

#include <memory>
#include <libdevcrypto/Common.h>
#include <libdevcrypto/ECDHE.h>
#include <libdevcrypto/CryptoPP.h>
#include "RLPxFrameIO.h"
#include "Common.h"
namespace ba = boost::asio;
namespace bi = boost::asio::ip;

namespace dev
{
namespace p2p
{

class RLPXHandshake: public std::enable_shared_from_this<RLPXHandshake>
{
	friend class RLPXFrameIO;
	enum State
	{
		Error = -1,
		New,				// New->AckAuth				[egress: tx auth, ingress: rx auth]
		AckAuth,			// AckAuth->WriteHello		[egress: rx ack, ingress: tx ack]
		WriteHello,		// WriteHello				[tx caps, rx caps, writehello]
		ReadHello,
		StartSession
	};
	
public:
	/// Handshake for ingress connection. Takes ownership of socket.
	RLPXHandshake(Host* _host, std::shared_ptr<RLPXSocket> const& _socket): m_host(_host), m_originated(false), m_socket(_socket) { crypto::Nonce::get().ref().copyTo(m_nonce.ref()); }
	
	/// Handshake for egress connection to _remote. Takes ownership of socket.
	RLPXHandshake(Host* _host, std::shared_ptr<RLPXSocket> const& _socket, NodeId _remote): m_host(_host), m_remote(_remote), m_originated(true), m_socket(_socket) { crypto::Nonce::get().ref().copyTo(m_nonce.ref()); }

	~RLPXHandshake() {}

	void start() { transition(); }

protected:
	void writeAuth();
	void readAuth();
	
	void writeAck();
	void readAck();
	
	void error();
	void transition(boost::system::error_code _ech = boost::system::error_code());

	/// Current state of handshake.
	State m_nextState = New;
	
	Host* m_host;
	
	/// Node id of remote host for socket.
	NodeId m_remote;
	bool m_originated = false;
	
	/// Buffers for encoded and decoded handshake phases
	bytes m_auth;
	bytes m_authCipher;
	bytes m_ack;
	bytes m_ackCipher;
	bytes m_handshakeOutBuffer;
	bytes m_handshakeInBuffer;
	
	crypto::ECDHE m_ecdhe;
	h256 m_nonce;
	
	Public m_remoteEphemeral;
	h256 m_remoteNonce;
	
	/// Frame IO is used to read frame for last step of handshake authentication.
	std::unique_ptr<RLPXFrameIO> m_io;
	std::shared_ptr<RLPXSocket> m_socket;
};
	
}
}