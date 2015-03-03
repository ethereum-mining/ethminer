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

RLPXFrameIO::RLPXFrameIO(bool _originated, Secret const& _ephemeralShared, bytesConstRef _authCipher, bytesConstRef _ackCipher): m_keys(h128(), h128()), m_macUpdateEncryptor(sha3("test").data(), 16)
{
	// we need:
	// originated?
	// Secret == output of ecdhe agreement
	// authCipher
	// ackCipher

	bytes keyMaterialBytes(512);
	bytesRef keyMaterial(&keyMaterialBytes);

//	ecdhe.agree(remoteEphemeral, ess);
	_ephemeralShared.ref().copyTo(keyMaterial.cropped(0, h256::size));
//	ss.ref().copyTo(keyMaterial.cropped(h256::size, h256::size));
//	//		auto token = sha3(ssA);
//	k->encryptK = sha3(keyMaterial);
//	k->encryptK.ref().copyTo(keyMaterial.cropped(h256::size, h256::size));
//	k->macK = sha3(keyMaterial);
//	
//	// Initiator egress-mac: sha3(mac-secret^recipient-nonce || auth-sent-init)
//	//           ingress-mac: sha3(mac-secret^initiator-nonce || auth-recvd-ack)
//	// Recipient egress-mac: sha3(mac-secret^initiator-nonce || auth-sent-ack)
//	//           ingress-mac: sha3(mac-secret^recipient-nonce || auth-recvd-init)
//	
//	bytes const& egressCipher = _originated ? authCipher : ackCipher;
//	keyMaterialBytes.resize(h256::size + egressCipher.size());
//	keyMaterial.retarget(keyMaterialBytes.data(), keyMaterialBytes.size());
//	(k->macK ^ remoteNonce).ref().copyTo(keyMaterial);
//	bytesConstRef(&egressCipher).copyTo(keyMaterial.cropped(h256::size, egressCipher.size()));
//	k->egressMac = sha3(keyMaterial);
//	
//	bytes const& ingressCipher = _originated ? ackCipher : authCipher;
//	keyMaterialBytes.resize(h256::size + ingressCipher.size());
//	keyMaterial.retarget(keyMaterialBytes.data(), keyMaterialBytes.size());
//	(k->macK ^ nonce).ref().copyTo(keyMaterial);
//	bytesConstRef(&ingressCipher).copyTo(keyMaterial.cropped(h256::size, ingressCipher.size()));
//	k->ingressMac = sha3(keyMaterial);

}

void RLPXFrameIO::writeFullPacketFrame(bytesConstRef _packet)
{
	
}

void RLPXFrameIO::writeHeader(bi::tcp::socket* _socket, h128 const& _header)
{
	
}

void RLPXFrameIO::write(bi::tcp::socket* _socket, bytesConstRef _in, bool _eof)
{
	
}

bool RLPXFrameIO::read(bytesConstRef _in, bytes& o_out)
{
	
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

void RLPXFrameIO::updateEgressMACWithHeader(h128 const& _headerCipher)
{
	m_egressMac.Update(_headerCipher.data(), h128::size);
	updateMAC(m_egressMac, *(h128*)_headerCipher.data());
}

void RLPXFrameIO::updateEgressMACWithEndOfFrame(bytesConstRef _cipher)
{
	m_egressMac.Update(_cipher.data(), _cipher.size());
	updateMAC(m_egressMac);
}

void RLPXFrameIO::updateIngressMACWithHeader(bytesConstRef _headerCipher)
{
	m_ingressMac.Update(_headerCipher.data(), h128::size);
	updateMAC(m_ingressMac, *(h128*)_headerCipher.data());
}

void RLPXFrameIO::updateIngressMACWithEndOfFrame(bytesConstRef _cipher)
{
	m_ingressMac.Update(_cipher.data(), _cipher.size());
	updateMAC(m_ingressMac);
}

void RLPXFrameIO::updateMAC(SHA3_256& _mac, h128 const& _seed)
{
	SHA3_256 prevDigest(_mac);
	h128 prevDigestOut;
	prevDigest.TruncatedFinal(prevDigestOut.data(), h128::size);
	
	h128 encDigest;
	m_macUpdateEncryptor.ProcessData(encDigest.data(), prevDigestOut.data(), h128::size);
	encDigest ^= (!!_seed ? _seed : prevDigestOut);
	
	// update mac for final digest
	_mac.Update(encDigest.data(), h256::size);
}


void RLPXHandshake::generateAuth()
{
	auth.resize(Signature::size + h256::size + Public::size + h256::size + 1);
	bytesRef sig(&auth[0], Signature::size);
	bytesRef hepubk(&auth[Signature::size], h256::size);
	bytesRef pubk(&auth[Signature::size + h256::size], Public::size);
	bytesRef nonce(&auth[Signature::size + h256::size + Public::size], h256::size);
	
	// E(remote-pubk, S(ecdhe-random, ecdh-shared-secret^nonce) || H(ecdhe-random-pubk) || pubk || nonce || 0x0)
	crypto::ecdh::agree(host->m_alias.sec(), remote, ss);
	sign(ecdhe.seckey(), ss ^ this->nonce).ref().copyTo(sig);
	sha3(ecdhe.pubkey().ref(), hepubk);
	host->m_alias.pub().ref().copyTo(pubk);
	this->nonce.ref().copyTo(nonce);
	auth[auth.size() - 1] = 0x0;
	encryptECIES(remote, &auth, authCipher);
}

void RLPXHandshake::generateAck()
{
	
}

bool RLPXHandshake::decodeAuth()
{
	if (!decryptECIES(host->m_alias.sec(), bytesConstRef(&authCipher), auth))
		return false;
	
	bytesConstRef sig(&auth[0], Signature::size);
	bytesConstRef hepubk(&auth[Signature::size], h256::size);
	bytesConstRef pubk(&auth[Signature::size + h256::size], Public::size);
	bytesConstRef nonce(&auth[Signature::size + h256::size + Public::size], h256::size);
	pubk.copyTo(remote.ref());
	nonce.copyTo(remoteNonce.ref());
	
	crypto::ecdh::agree(host->m_alias.sec(), remote, ss);
	remoteEphemeral = recover(*(Signature*)sig.data(), ss ^ remoteNonce);
	assert(sha3(remoteEphemeral) == *(h256*)hepubk.data());
	return true;
}

bool RLPXHandshake::decodeAck()
{
	
}

/// used for protocol handshake
bytes RLPXHandshake::frame(bytesConstRef _packet)
{
	
}

void RLPXHandshake::transition(boost::system::error_code _ech)
{
	if (_ech || nextState == Error)
	{
		clog(NetConnect) << "Disconnecting " << socket->remote_endpoint() << " (Handshake Failed)";
		boost::system::error_code ec;
		socket->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
		if (socket->is_open())
			socket->close();
		return;
	}
	
	auto self(shared_from_this());
	if (nextState == New)
	{
		nextState = AckAuth;
		
		clog(NetConnect) << "Authenticating connection for " << socket->remote_endpoint();
		
		if (originated)
		{
			clog(NetConnect) << "p2p.connect.egress sending auth";
			generateAuth();
			ba::async_write(*socket, ba::buffer(authCipher), [this, self](boost::system::error_code ec, std::size_t)
			{
				transition(ec);
			});
		}
		else
		{
			clog(NetConnect) << "p2p.connect.ingress recving auth";
			authCipher.resize(321);
			ba::async_read(*socket, ba::buffer(authCipher, 321), [this, self](boost::system::error_code ec, std::size_t)
			{
				if (ec)
					transition(ec);
				else if (decodeAuth())
					transition();
				else
				{
					clog(NetWarn) << "p2p.connect.egress recving auth decrypt failed";
					nextState = Error;
					transition();
					return;
				}
			});
		}
	}
	else if (nextState == AckAuth)
	{
		nextState = Authenticating;
		
		if (originated)
		{
			clog(NetConnect) << "p2p.connect.egress recving ack";
			// egress: rx ack
			ackCipher.resize(225);
			ba::async_read(*socket, ba::buffer(ackCipher, 225), [this, self](boost::system::error_code ec, std::size_t)
			{
				if (ec)
					transition(ec);
				else
				{
					if (!decryptECIES(host->m_alias.sec(), bytesConstRef(&ackCipher), ack))
					{
						clog(NetWarn) << "p2p.connect.egress recving ack decrypt failed";
						nextState = Error;
						transition();
						return;
					}
					
					bytesConstRef(&ack).cropped(0, Public::size).copyTo(remoteEphemeral.ref());
					bytesConstRef(&ack).cropped(Public::size, h256::size).copyTo(remoteNonce.ref());
					transition();
				}
			});
		}
		else
		{
			clog(NetConnect) << "p2p.connect.ingress sending ack";
			// ingress: tx ack
			ack.resize(Public::size + h256::size + 1);
			bytesRef epubk(&ack[0], Public::size);
			bytesRef nonce(&ack[Public::size], h256::size);
			ecdhe.pubkey().ref().copyTo(epubk);
			this->nonce.ref().copyTo(nonce);
			ack[ack.size() - 1] = 0x0;
			encryptECIES(remote, &ack, ackCipher);
			ba::async_write(*socket, ba::buffer(ackCipher), [this, self](boost::system::error_code ec, std::size_t)
			{
				transition(ec);
			});
		}
	}
	else if (nextState == Authenticating)
	{
		if (originated)
			clog(NetConnect) << "p2p.connect.egress sending magic sequence";
		else
			clog(NetConnect) << "p2p.connect.ingress sending magic sequence";
		PeerSecrets* k = new PeerSecrets;
		bytes keyMaterialBytes(512);
		bytesRef keyMaterial(&keyMaterialBytes);

		ecdhe.agree(remoteEphemeral, ess);
		ess.ref().copyTo(keyMaterial.cropped(0, h256::size));
		ss.ref().copyTo(keyMaterial.cropped(h256::size, h256::size));
		//		auto token = sha3(ssA);
		k->encryptK = sha3(keyMaterial);
		k->encryptK.ref().copyTo(keyMaterial.cropped(h256::size, h256::size));
		k->macK = sha3(keyMaterial);
		
		// Initiator egress-mac: sha3(mac-secret^recipient-nonce || auth-sent-init)
		//           ingress-mac: sha3(mac-secret^initiator-nonce || auth-recvd-ack)
		// Recipient egress-mac: sha3(mac-secret^initiator-nonce || auth-sent-ack)
		//           ingress-mac: sha3(mac-secret^recipient-nonce || auth-recvd-init)
		
		bytes const& egressCipher = originated ? authCipher : ackCipher;
		keyMaterialBytes.resize(h256::size + egressCipher.size());
		keyMaterial.retarget(keyMaterialBytes.data(), keyMaterialBytes.size());
		(k->macK ^ remoteNonce).ref().copyTo(keyMaterial);
		bytesConstRef(&egressCipher).copyTo(keyMaterial.cropped(h256::size, egressCipher.size()));
		k->egressMac = sha3(keyMaterial);
		
		bytes const& ingressCipher = originated ? ackCipher : authCipher;
		keyMaterialBytes.resize(h256::size + ingressCipher.size());
		keyMaterial.retarget(keyMaterialBytes.data(), keyMaterialBytes.size());
		(k->macK ^ nonce).ref().copyTo(keyMaterial);
		bytesConstRef(&ingressCipher).copyTo(keyMaterial.cropped(h256::size, ingressCipher.size()));
		k->ingressMac = sha3(keyMaterial);
		
		// This test will be replaced with protocol-capabilities information (was Hello packet)
		// TESTING: send encrypt magic sequence
		bytes magic {0x22,0x40,0x08,0x91};
		// rlpx encrypt
		encryptSymNoAuth(k->encryptK, &magic, k->magicCipherAndMac, h128());
		k->magicCipherAndMac.resize(k->magicCipherAndMac.size() + 32);
		sha3mac(k->egressMac.ref(), &magic, k->egressMac.ref());
		k->egressMac.ref().copyTo(bytesRef(&k->magicCipherAndMac).cropped(k->magicCipherAndMac.size() - 32, 32));
		
		clog(NetConnect) << "p2p.connect.egress txrx magic sequence";
		k->recvdMagicCipherAndMac.resize(k->magicCipherAndMac.size());
		
		ba::async_write(*socket, ba::buffer(k->magicCipherAndMac), [this, self, k, magic](boost::system::error_code ec, std::size_t)
		{
			if (ec)
			{
				delete k;
				transition(ec);
				return;
			}
			
			ba::async_read(*socket, ba::buffer(k->recvdMagicCipherAndMac, k->magicCipherAndMac.size()), [this, self, k, magic](boost::system::error_code ec, std::size_t)
			{
				if (originated)
					clog(NetNote) << "p2p.connect.egress recving magic sequence";
				else
					clog(NetNote) << "p2p.connect.ingress recving magic sequence";
				
				if (ec)
				{
					delete k;
					transition(ec);
					return;
				}
				
				/// capabilities handshake (encrypted magic sequence is placeholder)
				bytes decryptedMagic;
				decryptSymNoAuth(k->encryptK, h128(), &k->recvdMagicCipherAndMac, decryptedMagic);
				if (decryptedMagic[0] == 0x22 && decryptedMagic[1] == 0x40 && decryptedMagic[2] == 0x08 && decryptedMagic[3] == 0x91)
				{
					shared_ptr<Peer> p;
					p = host->m_peers[remote];
					if (!p)
					{
						p.reset(new Peer());
						p->id = remote;
					}
					p->endpoint.tcp.address(socket->remote_endpoint().address());
					p->m_lastDisconnect = NoDisconnect;
					p->m_lastConnected = std::chrono::system_clock::now();
					p->m_failedAttempts = 0;

					auto ps = std::make_shared<Session>(host, move(*socket), p);
					ps->start();
				}
				
				// todo: PeerSession will take ownership of k and use it to encrypt wireline.
				delete k;
			});
		});
	}
}
