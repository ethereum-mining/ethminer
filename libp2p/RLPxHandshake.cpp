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
/** @file RLPXHandshake.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2015
 */

#include "Host.h"
#include "Session.h"
#include "Peer.h"
#include "RLPxHandshake.h"
using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace CryptoPP;

void RLPXHandshake::writeAuth()
{
	clog(NetP2PConnect) << "p2p.connect.egress sending auth to " << m_socket->remoteEndpoint();
	m_auth.resize(Signature::size + h256::size + Public::size + h256::size + 1);
	bytesRef sig(&m_auth[0], Signature::size);
	bytesRef hepubk(&m_auth[Signature::size], h256::size);
	bytesRef pubk(&m_auth[Signature::size + h256::size], Public::size);
	bytesRef nonce(&m_auth[Signature::size + h256::size + Public::size], h256::size);
	
	// E(remote-pubk, S(ecdhe-random, ecdh-shared-secret^nonce) || H(ecdhe-random-pubk) || pubk || nonce || 0x0)
	Secret staticShared;
	crypto::ecdh::agree(m_host->m_alias.sec(), m_remote, staticShared);
	sign(m_ecdhe.seckey(), staticShared.makeInsecure() ^ m_nonce).ref().copyTo(sig);
	sha3(m_ecdhe.pubkey().ref(), hepubk);
	m_host->m_alias.pub().ref().copyTo(pubk);
	m_nonce.ref().copyTo(nonce);
	m_auth[m_auth.size() - 1] = 0x0;
	encryptECIES(m_remote, &m_auth, m_authCipher);

	auto self(shared_from_this());
	ba::async_write(m_socket->ref(), ba::buffer(m_authCipher), [this, self](boost::system::error_code ec, std::size_t)
	{
		transition(ec);
	});
}

void RLPXHandshake::writeAck()
{
	clog(NetP2PConnect) << "p2p.connect.ingress sending ack to " << m_socket->remoteEndpoint();
	m_ack.resize(Public::size + h256::size + 1);
	bytesRef epubk(&m_ack[0], Public::size);
	bytesRef nonce(&m_ack[Public::size], h256::size);
	m_ecdhe.pubkey().ref().copyTo(epubk);
	m_nonce.ref().copyTo(nonce);
	m_ack[m_ack.size() - 1] = 0x0;
	encryptECIES(m_remote, &m_ack, m_ackCipher);
	
	auto self(shared_from_this());
	ba::async_write(m_socket->ref(), ba::buffer(m_ackCipher), [this, self](boost::system::error_code ec, std::size_t)
	{
		transition(ec);
	});
}

void RLPXHandshake::readAuth()
{
	clog(NetP2PConnect) << "p2p.connect.ingress recving auth from " << m_socket->remoteEndpoint();
	m_authCipher.resize(307);
	auto self(shared_from_this());
	ba::async_read(m_socket->ref(), ba::buffer(m_authCipher, 307), [this, self](boost::system::error_code ec, std::size_t)
	{
		if (ec)
			transition(ec);
		else if (decryptECIES(m_host->m_alias.sec(), bytesConstRef(&m_authCipher), m_auth))
		{
			bytesConstRef sig(&m_auth[0], Signature::size);
			bytesConstRef hepubk(&m_auth[Signature::size], h256::size);
			bytesConstRef pubk(&m_auth[Signature::size + h256::size], Public::size);
			bytesConstRef nonce(&m_auth[Signature::size + h256::size + Public::size], h256::size);
			pubk.copyTo(m_remote.ref());
			nonce.copyTo(m_remoteNonce.ref());
			
			Secret sharedSecret;
			crypto::ecdh::agree(m_host->m_alias.sec(), m_remote, sharedSecret);
			m_remoteEphemeral = recover(*(Signature*)sig.data(), sharedSecret.makeInsecure() ^ m_remoteNonce);

			if (sha3(m_remoteEphemeral) != *(h256*)hepubk.data())
				clog(NetP2PConnect) << "p2p.connect.ingress auth failed (invalid: hash mismatch) for" << m_socket->remoteEndpoint();
			
			transition();
		}
		else
		{
			clog(NetP2PConnect) << "p2p.connect.ingress recving auth decrypt failed for" << m_socket->remoteEndpoint();
			m_nextState = Error;
			transition();
		}
	});
}

void RLPXHandshake::readAck()
{
	clog(NetP2PConnect) << "p2p.connect.egress recving ack from " << m_socket->remoteEndpoint();
	m_ackCipher.resize(210);
	auto self(shared_from_this());
	ba::async_read(m_socket->ref(), ba::buffer(m_ackCipher, 210), [this, self](boost::system::error_code ec, std::size_t)
	{
		if (ec)
			transition(ec);
		else if (decryptECIES(m_host->m_alias.sec(), bytesConstRef(&m_ackCipher), m_ack))
		{
			bytesConstRef(&m_ack).cropped(0, Public::size).copyTo(m_remoteEphemeral.ref());
			bytesConstRef(&m_ack).cropped(Public::size, h256::size).copyTo(m_remoteNonce.ref());
			transition();
		}
		else
		{
			clog(NetP2PConnect) << "p2p.connect.egress recving ack decrypt failed for " << m_socket->remoteEndpoint();
			m_nextState = Error;
			transition();
		}
	});
}

void RLPXHandshake::error()
{
	m_idleTimer.cancel();
	
	auto connected = m_socket->isConnected();
	if (connected && !m_socket->remoteEndpoint().address().is_unspecified())
		clog(NetP2PConnect) << "Disconnecting " << m_socket->remoteEndpoint() << " (Handshake Failed)";
	else
		clog(NetP2PConnect) << "Handshake Failed (Connection reset by peer)";

	m_socket->close();
	if (m_io != nullptr)
		delete m_io;
}

void RLPXHandshake::transition(boost::system::error_code _ech)
{
	// reset timeout
	m_idleTimer.cancel();
	
	if (_ech || m_nextState == Error || m_cancel)
	{
		clog(NetP2PConnect) << "Handshake Failed (I/O Error:" << _ech.message() << ")";
		return error();
	}
	
	auto self(shared_from_this());
	assert(m_nextState != StartSession);
	m_idleTimer.expires_from_now(c_timeout);
	m_idleTimer.async_wait([this, self](boost::system::error_code const& _ec)
	{
		if (!_ec)
		{
			if (!m_socket->remoteEndpoint().address().is_unspecified())
				clog(NetP2PConnect) << "Disconnecting " << m_socket->remoteEndpoint() << " (Handshake Timeout)";
			cancel();
		}
	});
	
	if (m_nextState == New)
	{
		m_nextState = AckAuth;
		if (m_originated)
			writeAuth();
		else
			readAuth();
	}
	else if (m_nextState == AckAuth)
	{
		m_nextState = WriteHello;
		if (m_originated)
			readAck();
		else
			writeAck();
	}
	else if (m_nextState == WriteHello)
	{
		m_nextState = ReadHello;
		clog(NetP2PConnect) << (m_originated ? "p2p.connect.egress" : "p2p.connect.ingress") << "sending capabilities handshake";

		/// This pointer will be freed if there is an error otherwise
		/// it will be passed to Host which will take ownership.
		m_io = new RLPXFrameCoder(*this);

		// old packet format
		// 5 arguments, HelloPacket
		RLPStream s;
		s.append((unsigned)HelloPacket).appendList(5)
		<< dev::p2p::c_protocolVersion
		<< m_host->m_clientVersion
		<< m_host->caps()
		<< m_host->listenPort()
		<< m_host->id();
		bytes packet;
		s.swapOut(packet);
		m_io->writeSingleFramePacket(&packet, m_handshakeOutBuffer);
		ba::async_write(m_socket->ref(), ba::buffer(m_handshakeOutBuffer), [this, self](boost::system::error_code ec, std::size_t)
		{
			transition(ec);
		});
	}
	else if (m_nextState == ReadHello)
	{
		// Authenticate and decrypt initial hello frame with initial RLPXFrameCoder
		// and request m_host to start session.
		m_nextState = StartSession;
		
		// read frame header
		unsigned const handshakeSize = 32;
		m_handshakeInBuffer.resize(handshakeSize);
		ba::async_read(m_socket->ref(), boost::asio::buffer(m_handshakeInBuffer, handshakeSize), [this, self](boost::system::error_code ec, std::size_t)
		{
			if (ec)
				transition(ec);
			else
			{
				/// authenticate and decrypt header
				if (!m_io->authAndDecryptHeader(bytesRef(m_handshakeInBuffer.data(), m_handshakeInBuffer.size())))
				{
					m_nextState = Error;
					transition();
					return;
				}
				
				clog(NetP2PNote) << (m_originated ? "p2p.connect.egress" : "p2p.connect.ingress") << "recvd hello header";
				
				/// check frame size
				bytes& header = m_handshakeInBuffer;
				uint32_t frameSize = (uint32_t)(header[2]) | (uint32_t)(header[1])<<8 | (uint32_t)(header[0])<<16;
				if (frameSize > 1024)
				{
					// all future frames: 16777216
					clog(NetP2PWarn) << (m_originated ? "p2p.connect.egress" : "p2p.connect.ingress") << "hello frame is too large" << frameSize;
					m_nextState = Error;
					transition();
					return;
				}
				
				/// rlp of header has protocol-type, sequence-id[, total-packet-size]
				bytes headerRLP(header.size() - 3 - h128::size);	// this is always 32 - 3 - 16 = 13. wtf?
				bytesConstRef(&header).cropped(3).copyTo(&headerRLP);
				
				/// read padded frame and mac
				m_handshakeInBuffer.resize(frameSize + ((16 - (frameSize % 16)) % 16) + h128::size);
				ba::async_read(m_socket->ref(), boost::asio::buffer(m_handshakeInBuffer, m_handshakeInBuffer.size()), [this, self, headerRLP](boost::system::error_code ec, std::size_t)
				{
					m_idleTimer.cancel();
					
					if (ec)
						transition(ec);
					else
					{
						bytesRef frame(&m_handshakeInBuffer);
						if (!m_io->authAndDecryptFrame(frame))
						{
							clog(NetTriviaSummary) << (m_originated ? "p2p.connect.egress" : "p2p.connect.ingress") << "hello frame: decrypt failed";
							m_nextState = Error;
							transition();
							return;
						}
						
						PacketType packetType = frame[0] == 0x80 ? HelloPacket : (PacketType)frame[0];
						if (packetType != HelloPacket)
						{
							clog(NetTriviaSummary) << (m_originated ? "p2p.connect.egress" : "p2p.connect.ingress") << "hello frame: invalid packet type";
							m_nextState = Error;
							transition();
							return;
						}

						clog(NetTriviaSummary) << (m_originated ? "p2p.connect.egress" : "p2p.connect.ingress") << "hello frame: success. starting session.";
						try
						{
							RLP rlp(frame.cropped(1), RLP::ThrowOnFail | RLP::FailIfTooSmall);
							m_host->startPeerSession(m_remote, rlp, m_io, m_socket);
						}
						catch (std::exception const& _e)
						{
							clog(NetWarn) << "Handshake causing an exception:" << _e.what();
							m_nextState = Error;
							transition();
						}
					}
				});
			}
		});
	}
}
