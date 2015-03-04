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
#include "Common.h"
namespace ba = boost::asio;
namespace bi = boost::asio::ip;

namespace dev
{
namespace p2p
{

class RLPXHandshake;

class RLPXFrameIO
{
public:
	RLPXFrameIO(RLPXHandshake& _init);
	
	void writeFullPacketFrame(bytesConstRef _packet);
	
	void writeHeader(bi::tcp::socket* _socket, h128 const& _header);
	
	void write(bi::tcp::socket* _socket, bytesConstRef _in, bool _eof = false);
	
	bool read(bytesConstRef _in, bytes& o_out);
	
	h128 egressDigest();
	
	h128 ingressDigest();
	
	void updateEgressMACWithHeader(h128 const& _headerCipher);
	
	void updateEgressMACWithEndOfFrame(bytesConstRef _cipher);
	
	void updateIngressMACWithHeader(bytesConstRef _headerCipher);
	
	void updateIngressMACWithEndOfFrame(bytesConstRef _cipher);
	
private:
	void updateMAC(CryptoPP::SHA3_256& _mac, h128 const& _seed = h128());

	CryptoPP::CTR_Mode<CryptoPP::AES>::Encryption m_frameEnc;
	CryptoPP::ECB_Mode<CryptoPP::AES>::Encryption m_macEnc;
	CryptoPP::SHA3_256 m_egressMac;
	CryptoPP::SHA3_256 m_ingressMac;
};

struct RLPXHandshake: public std::enable_shared_from_this<RLPXHandshake>
{
	friend class RLPXFrameIO;
	friend class Host;
	enum State
	{
		Error = -1,
		New,				// New->AckAuth				[egress: tx auth, ingress: rx auth]
		AckAuth,			// AckAuth->Authenticating	[egress: rx ack, ingress: tx ack]
		Authenticating,	// Authenticating			[tx caps, rx caps, authenticate]
	};

	/// Handshake for ingress connection. Takes ownership of socket.
	RLPXHandshake(Host* _host, bi::tcp::socket* _socket): host(_host), socket(std::move(_socket)), originated(false) { crypto::Nonce::get().ref().copyTo(nonce.ref()); }
	
	/// Handshake for egress connection to _remote. Takes ownership of socket.
	RLPXHandshake(Host* _host, bi::tcp::socket* _socket, NodeId _remote): host(_host), remote(_remote), socket(std::move(_socket)), originated(true) { crypto::Nonce::get().ref().copyTo(nonce.ref()); }
	
	~RLPXHandshake() { delete socket; }

protected:
	void start() { transition(); }
	
	void generateAuth();
	bool decodeAuth();
	
	void generateAck();
	bool decodeAck();
	
	bytes frame(bytesConstRef _packet);

private:
	void transition(boost::system::error_code _ech = boost::system::error_code());
	
	/// Current state of handshake.
	State nextState = New;
	
	Host* host;
	
	/// Node id of remote host for socket.
	NodeId remote;
	
	bi::tcp::socket* socket;
	bool originated = false;
	
	/// Buffers for encoded and decoded handshake phases
	bytes auth;
	bytes authCipher;
	bytes ack;
	bytes ackCipher;

	crypto::ECDHE ecdhe;
	h256 nonce;
	
	Public remoteEphemeral;
	h256 remoteNonce;
};
	
}
}