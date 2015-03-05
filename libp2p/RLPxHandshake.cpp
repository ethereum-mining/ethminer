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

RLPXFrameIO::RLPXFrameIO(RLPXHandshake const& _init): m_socket(_init.socket)
{
	// we need:
	// originated?
	// Secret == output of ecdhe agreement
	// authCipher
	// ackCipher

	bytes keyMaterialBytes(64);
	bytesRef keyMaterial(&keyMaterialBytes);

	// shared-secret = sha3(ecdhe-shared-secret || sha3(nonce || initiator-nonce))
	Secret ephemeralShared;
	_init.ecdhe.agree(_init.remoteEphemeral, ephemeralShared);
	ephemeralShared.ref().copyTo(keyMaterial.cropped(0, h256::size));
	h512 nonceMaterial;
	h256 const& leftNonce = _init.originated ? _init.remoteNonce : _init.nonce;
	h256 const& rightNonce = _init.originated ? _init.nonce : _init.remoteNonce;
	leftNonce.ref().copyTo(nonceMaterial.ref().cropped(0, h256::size));
	rightNonce.ref().copyTo(nonceMaterial.ref().cropped(h256::size, h256::size));
	auto outRef(keyMaterial.cropped(h256::size, h256::size));
	sha3(nonceMaterial.ref(), outRef); // output h(nonces)
	
	sha3(keyMaterial, outRef); // output shared-secret
	// token: sha3(outRef, bytesRef(&token)); -> Host (to be saved to disk)
	
	// aes-secret = sha3(ecdhe-shared-secret || shared-secret)
	sha3(keyMaterial, outRef); // output aes-secret
	m_frameEnc.SetKeyWithIV(outRef.data(), h128::size, h128().data());
	m_frameDec.SetKeyWithIV(outRef.data(), h128::size, h128().data());

	// mac-secret = sha3(ecdhe-shared-secret || aes-secret)
	sha3(keyMaterial, outRef); // output mac-secret
	m_macEnc.SetKey(outRef.data(), h128::size);

	// Initiator egress-mac: sha3(mac-secret^recipient-nonce || auth-sent-init)
	//           ingress-mac: sha3(mac-secret^initiator-nonce || auth-recvd-ack)
	// Recipient egress-mac: sha3(mac-secret^initiator-nonce || auth-sent-ack)
	//           ingress-mac: sha3(mac-secret^recipient-nonce || auth-recvd-init)
 
	(*(h256*)outRef.data() ^ _init.remoteNonce).ref().copyTo(keyMaterial);
	bytes const& egressCipher = _init.originated ? _init.authCipher : _init.ackCipher;
	keyMaterialBytes.resize(h256::size + egressCipher.size());
	keyMaterial.retarget(keyMaterialBytes.data(), keyMaterialBytes.size());
	bytesConstRef(&egressCipher).copyTo(keyMaterial.cropped(h256::size, egressCipher.size()));
	m_egressMac.Update(keyMaterial.data(), keyMaterial.size());

	// recover mac-secret by re-xoring remoteNonce
	(*(h256*)keyMaterial.data() ^ _init.remoteNonce ^ _init.nonce).ref().copyTo(keyMaterial);
	bytes const& ingressCipher = _init.originated ? _init.ackCipher : _init.authCipher;
	keyMaterialBytes.resize(h256::size + ingressCipher.size());
	keyMaterial.retarget(keyMaterialBytes.data(), keyMaterialBytes.size());
	bytesConstRef(&ingressCipher).copyTo(keyMaterial.cropped(h256::size, ingressCipher.size()));
	m_ingressMac.Update(keyMaterial.data(), keyMaterial.size());
}

void RLPXFrameIO::writeSingleFramePacket(bytesConstRef _packet, bytes& o_bytes)
{
	// _packet = type || rlpList()

	// current/old packet format: prep(_s).appendList(_args + 1).append((unsigned)_id);
	RLPStream header;
	header.appendRaw(bytes({byte(_packet.size() >> 16), byte(_packet.size() >> 8), byte(_packet.size())}));
	// zeroHeader: []byte{0xC2, 0x80, 0x80}. Should be rlpList(protocolType,seqId,totalPacketSize).
	header.appendRaw(bytes({0xc2,0x80,0x80}));
	
	// TODO: SECURITY check that header is <= 16 bytes
	
	bytes headerWithMac;
	header.swapOut(headerWithMac);
	headerWithMac.resize(32);
	m_frameEnc.ProcessData(headerWithMac.data(), headerWithMac.data(), 16);
	updateEgressMACWithHeader(bytesConstRef(&headerWithMac).cropped(0, 16));
	egressDigest().ref().copyTo(bytesRef(&headerWithMac).cropped(h128::size,h128::size));

	auto padding = (16 - (_packet.size() % 16)) % 16;
	o_bytes.swap(headerWithMac);
	o_bytes.resize(32 + _packet.size() + padding + h128::size);
	bytesRef packetRef(o_bytes.data() + 32, _packet.size());
	m_frameEnc.ProcessData(packetRef.data(), _packet.data(), _packet.size());
	bytesRef paddingRef(o_bytes.data() + 32 + _packet.size(), padding);
	if (padding)
		m_frameEnc.ProcessData(paddingRef.data(), paddingRef.data(), padding);
	bytesRef packetWithPaddingRef(o_bytes.data() + 32, _packet.size() + padding);
	updateEgressMACWithEndOfFrame(packetWithPaddingRef);
	bytesRef macRef(o_bytes.data() + 32 + _packet.size() + padding, h128::size);
	egressDigest().ref().copyTo(macRef);
	clog(NetConnect) << "SENT FRAME " << _packet.size() << *(h128*)macRef.data();
	clog(NetConnect) << "FRAME TAIL " << *(h128*)(o_bytes.data() + 32 + _packet.size() + padding);
}

bool RLPXFrameIO::authAndDecryptHeader(h256& io)
{
	updateIngressMACWithHeader(io.ref());
	bytesConstRef macRef = io.ref().cropped(h128::size, h128::size);
	if (*(h128*)macRef.data() != ingressDigest())
		return false;
	m_frameDec.ProcessData(io.data(), io.data(), 16);
	return true;
}

bool RLPXFrameIO::authAndDecryptFrame(bytesRef io)
{
	bytesRef cipherText(io.cropped(0, io.size() - h128::size));
	updateIngressMACWithEndOfFrame(cipherText);
	bytesConstRef frameMac(io.data() + io.size() - h128::size, h128::size);
	if (*(h128*)frameMac.data() != ingressDigest())
		return false;
	m_frameDec.ProcessData(io.data(), io.data(), io.size() - h128::size);
	return true;
}

h128 RLPXFrameIO::egressDigest()
{
	SHA3_256 h(m_egressMac);
	h128 digest;
	h.TruncatedFinal(digest.data(), h128::size);
	return move(digest);
}

h128 RLPXFrameIO::ingressDigest()
{
	SHA3_256 h(m_ingressMac);
	h128 digest;
	h.TruncatedFinal(digest.data(), h128::size);
	return move(digest);
}

void RLPXFrameIO::updateEgressMACWithHeader(bytesConstRef _headerCipher)
{
	updateMAC(m_egressMac, *(h128*)_headerCipher.data());
}

void RLPXFrameIO::updateEgressMACWithEndOfFrame(bytesConstRef _cipher)
{
	m_egressMac.Update(_cipher.data(), _cipher.size());
	updateMAC(m_egressMac);
	{
		SHA3_256 prev(m_egressMac);
		h128 digest;
		prev.TruncatedFinal(digest.data(), h128::size);
		clog(NetConnect) << "EGRESS FRAMEMAC " << _cipher.size() << digest;
	}
}

void RLPXFrameIO::updateIngressMACWithHeader(bytesConstRef _headerCipher)
{
	updateMAC(m_ingressMac, *(h128*)_headerCipher.data());
}

void RLPXFrameIO::updateIngressMACWithEndOfFrame(bytesConstRef _cipher)
{
	m_ingressMac.Update(_cipher.data(), _cipher.size());
	updateMAC(m_ingressMac);
	{
		SHA3_256 prev(m_ingressMac);
		h128 digest;
		prev.TruncatedFinal(digest.data(), h128::size);
		clog(NetConnect) << "INGRESS FRAMEMAC " << _cipher.size() << digest;
	}
}

void RLPXFrameIO::updateMAC(SHA3_256& _mac, h128 const& _seed)
{
	SHA3_256 prevDigest(_mac);
	h128 prevDigestOut;
	prevDigest.TruncatedFinal(prevDigestOut.data(), h128::size);
	
	h128 encDigest;
	m_macEnc.ProcessData(encDigest.data(), prevDigestOut.data(), h128::size);
	encDigest ^= (!!_seed ? _seed : prevDigestOut);
	
	// update mac for final digest
	_mac.Update(encDigest.data(), h128::size);
}


void RLPXHandshake::writeAuth()
{
	clog(NetConnect) << "p2p.connect.egress sending auth to " << socket->remote_endpoint();
	auth.resize(Signature::size + h256::size + Public::size + h256::size + 1);
	bytesRef sig(&auth[0], Signature::size);
	bytesRef hepubk(&auth[Signature::size], h256::size);
	bytesRef pubk(&auth[Signature::size + h256::size], Public::size);
	bytesRef nonce(&auth[Signature::size + h256::size + Public::size], h256::size);
	
	// E(remote-pubk, S(ecdhe-random, ecdh-shared-secret^nonce) || H(ecdhe-random-pubk) || pubk || nonce || 0x0)
	Secret staticShared;
	crypto::ecdh::agree(host->m_alias.sec(), remote, staticShared);
	sign(ecdhe.seckey(), staticShared ^ this->nonce).ref().copyTo(sig);
	sha3(ecdhe.pubkey().ref(), hepubk);
	host->m_alias.pub().ref().copyTo(pubk);
	this->nonce.ref().copyTo(nonce);
	auth[auth.size() - 1] = 0x0;
	encryptECIES(remote, &auth, authCipher);

	auto self(shared_from_this());
	ba::async_write(*socket, ba::buffer(authCipher), [this, self](boost::system::error_code ec, std::size_t)
	{
		transition(ec);
	});
}

void RLPXHandshake::writeAck()
{
	clog(NetConnect) << "p2p.connect.ingress sending ack to " << socket->remote_endpoint();
	ack.resize(Public::size + h256::size + 1);
	bytesRef epubk(&ack[0], Public::size);
	bytesRef nonce(&ack[Public::size], h256::size);
	ecdhe.pubkey().ref().copyTo(epubk);
	this->nonce.ref().copyTo(nonce);
	ack[ack.size() - 1] = 0x0;
	encryptECIES(remote, &ack, ackCipher);
	
	auto self(shared_from_this());
	ba::async_write(*socket, ba::buffer(ackCipher), [this, self](boost::system::error_code ec, std::size_t)
	{
		transition(ec);
	});
}

void RLPXHandshake::readAuth()
{
	clog(NetConnect) << "p2p.connect.ingress recving auth from " << socket->remote_endpoint();
	authCipher.resize(307);
	auto self(shared_from_this());
	ba::async_read(*socket, ba::buffer(authCipher, 307), [this, self](boost::system::error_code ec, std::size_t)
	{
		if (ec)
			transition(ec);
		else if (decryptECIES(host->m_alias.sec(), bytesConstRef(&authCipher), auth))
		{
			bytesConstRef sig(&auth[0], Signature::size);
			bytesConstRef hepubk(&auth[Signature::size], h256::size);
			bytesConstRef pubk(&auth[Signature::size + h256::size], Public::size);
			bytesConstRef nonce(&auth[Signature::size + h256::size + Public::size], h256::size);
			pubk.copyTo(remote.ref());
			nonce.copyTo(remoteNonce.ref());
			
			Secret sharedSecret;
			crypto::ecdh::agree(host->m_alias.sec(), remote, sharedSecret);
			remoteEphemeral = recover(*(Signature*)sig.data(), sharedSecret ^ remoteNonce);
			assert(sha3(remoteEphemeral) == *(h256*)hepubk.data());
			transition();
		}
		else
		{
			clog(NetWarn) << "p2p.connect.egress recving auth decrypt failed for" << socket->remote_endpoint();
			nextState = Error;
			transition();
		}
	});
}

void RLPXHandshake::readAck()
{
	clog(NetConnect) << "p2p.connect.egress recving ack from " << socket->remote_endpoint();
	ackCipher.resize(210);
	auto self(shared_from_this());
	ba::async_read(*socket, ba::buffer(ackCipher, 210), [this, self](boost::system::error_code ec, std::size_t)
	{
		if (ec)
			transition(ec);
		else if (decryptECIES(host->m_alias.sec(), bytesConstRef(&ackCipher), ack))
		{
			bytesConstRef(&ack).cropped(0, Public::size).copyTo(remoteEphemeral.ref());
			bytesConstRef(&ack).cropped(Public::size, h256::size).copyTo(remoteNonce.ref());
			transition();
		}
		else
		{
			clog(NetWarn) << "p2p.connect.egress recving ack decrypt failed for " << socket->remote_endpoint();
			nextState = Error;
			transition();
		}
	});
}

void RLPXHandshake::error()
{
	clog(NetConnect) << "Disconnecting " << socket->remote_endpoint() << " (Handshake Failed)";
	boost::system::error_code ec;
	socket->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
	if (socket->is_open())
		socket->close();
}

void RLPXHandshake::transition(boost::system::error_code _ech)
{
	if (_ech || nextState == Error)
		return error();
	
	auto self(shared_from_this());
	if (nextState == New)
	{
		nextState = AckAuth;
		if (originated)
			writeAuth();
		else
			readAuth();
	}
	else if (nextState == AckAuth)
	{
		nextState = WriteHello;
		if (originated)
			readAck();
		else
			writeAck();
	}
	else if (nextState == WriteHello)
	{
		nextState = ReadHello;
		
		if (originated)
			clog(NetConnect) << "p2p.connect.egress sending capabilities handshake";
		else
			clog(NetConnect) << "p2p.connect.ingress sending capabilities handshake";
		
		io = new RLPXFrameIO(*this);

		// old packet format
		// 5 arguments, HelloPacket
		RLPStream s;
		s.appendList(5 + 1).append((unsigned)0)
		<< host->protocolVersion()
		<< host->m_clientVersion
		<< host->caps()
		<< host->m_tcpPublic.port()
		<< host->id();
		bytes packet;
		s.swapOut(packet);
		io->writeSingleFramePacket(&packet, handshakeOutBuffer);
		ba::async_write(*socket, ba::buffer(handshakeOutBuffer), [this, self](boost::system::error_code ec, std::size_t)
		{
			transition(ec);
		});
	}
	else if (nextState == ReadHello)
	{
		// Authenticate and decrypt initial hello frame with initial RLPXFrameIO
		// and request host to start session.
		nextState = StartSession;
		
		// read frame header
		handshakeInBuffer.resize(h256::size);
		ba::async_read(*socket, boost::asio::buffer(handshakeInBuffer, h256::size), [this,self](boost::system::error_code ec, std::size_t length)
		{
			if (ec)
				transition(ec);
			else
			{
				/// authenticate and decrypt header
				if (!io->authAndDecryptHeader(*(h256*)handshakeInBuffer.data()))
				{
					nextState = Error;
					transition();
					return;
				}
				
				clog(NetNote) << (originated ? "p2p.connect.egress" : "p2p.connect.ingress") << "recvd hello header";
				
				/// check frame size
				bytes& header = handshakeInBuffer;
				uint32_t frameSize = (uint32_t)(header[2]) | (uint32_t)(header[1])<<8 | (uint32_t)(header[0])<<16;
				if (frameSize > 1024)
				{
					// all future frames: 16777216
					clog(NetWarn) << (originated ? "p2p.connect.egress" : "p2p.connect.ingress") << "hello frame is too large";
					nextState = Error;
					transition();
					return;
				}
				
				/// rlp of header has protocol-type, sequence-id[, total-packet-size]
				bytes headerRLP(header.size() - 3 - h128::size);
				bytesConstRef(&header).cropped(3).copyTo(&headerRLP);
				
				/// read padded frame and mac
				handshakeInBuffer.resize(frameSize + ((16 - (frameSize % 16)) % 16) + h128::size);
				ba::async_read(*socket, boost::asio::buffer(handshakeInBuffer, handshakeInBuffer.size()), [this, self, headerRLP](boost::system::error_code ec, std::size_t length)
				{
					if (ec)
						transition(ec);
					else
					{
						if (!io->authAndDecryptFrame(bytesRef(&handshakeInBuffer)))
						{
							clog(NetWarn) << (originated ? "p2p.connect.egress" : "p2p.connect.ingress") << "hello frame: decrypt failed";
							nextState = Error;
							transition();
							return;
						}
						
						RLP rlp(handshakeInBuffer);
						auto packetType = (PacketType)rlp[0].toInt<unsigned>();
						if (packetType != 0)
						{
							clog(NetWarn) << (originated ? "p2p.connect.egress" : "p2p.connect.ingress") << "hello frame: invalid packet type";
							nextState = Error;
							transition();
							return;
						}
						
						// todo: handover RLPFrameIO
						host->startPeerSession(remote, rlp, socket);
					}
				});
			}
		});
	}
}
