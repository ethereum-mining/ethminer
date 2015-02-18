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
/** @file Host.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <set>
#include <chrono>
#include <thread>
#include <mutex>
#include <memory>
#include <boost/algorithm/string.hpp>
#include <libdevcore/Common.h>
#include <libdevcore/CommonIO.h>
#include <libethcore/Exceptions.h>
#include <libdevcrypto/FileSystem.h>
#include "Session.h"
#include "Common.h"
#include "Capability.h"
#include "UPnP.h"
#include "Host.h"
using namespace std;
using namespace dev;
using namespace dev::p2p;

HostNodeTableHandler::HostNodeTableHandler(Host& _host): m_host(_host) {}

void HostNodeTableHandler::processEvent(NodeId const& _n, NodeTableEventType const& _e)
{
	m_host.onNodeTableEvent(_n, _e);
}

Host::Host(std::string const& _clientVersion, NetworkPreferences const& _n, bytesConstRef _restoreNetwork):
	Worker("p2p", 0),
	m_restoreNetwork(_restoreNetwork.toBytes()),
	m_clientVersion(_clientVersion),
	m_netPrefs(_n),
	m_ifAddresses(Network::getInterfaceAddresses()),
	m_ioService(2),
	m_tcp4Acceptor(m_ioService),
	m_alias(networkAlias(_restoreNetwork)),
	m_lastPing(chrono::steady_clock::time_point::min())
{
	for (auto address: m_ifAddresses)
		if (address.is_v4())
			clog(NetNote) << "IP Address: " << address << " = " << (isPrivateAddress(address) ? "[LOCAL]" : "[PEER]");
	
	clog(NetNote) << "Id:" << id();
}

Host::~Host()
{
	stop();
}

void Host::start()
{
	startWorking();
}

void Host::stop()
{
	// called to force io_service to kill any remaining tasks it might have -
	// such tasks may involve socket reads from Capabilities that maintain references
	// to resources we're about to free.

	{
		// Although m_run is set by stop() or start(), it effects m_runTimer so x_runTimer is used instead of a mutex for m_run.
		// when m_run == false, run() will cause this::run() to stop() ioservice
		Guard l(x_runTimer);
		// ignore if already stopped/stopping
		if (!m_run)
			return;
		m_run = false;
	}
	
	// wait for m_timer to reset (indicating network scheduler has stopped)
	while (!!m_timer)
		this_thread::sleep_for(chrono::milliseconds(50));
	
	// stop worker thread
	stopWorking();
}

void Host::doneWorking()
{
	// reset ioservice (allows manually polling network, below)
	m_ioService.reset();
	
	// shutdown acceptor
	m_tcp4Acceptor.cancel();
	if (m_tcp4Acceptor.is_open())
		m_tcp4Acceptor.close();
	
	// There maybe an incoming connection which started but hasn't finished.
	// Wait for acceptor to end itself instead of assuming it's complete.
	// This helps ensure a peer isn't stopped at the same time it's starting
	// and that socket for pending connection is closed.
	while (m_accepting)
		m_ioService.poll();

	// stop capabilities (eth: stops syncing or block/tx broadcast)
	for (auto const& h: m_capabilities)
		h.second->onStopping();

	// disconnect peers
	for (unsigned n = 0;; n = 0)
	{
		{
			RecursiveGuard l(x_sessions);
			for (auto i: m_sessions)
				if (auto p = i.second.lock())
					if (p->isConnected())
					{
						p->disconnect(ClientQuit);
						n++;
					}
		}
		if (!n)
			break;
		
		// poll so that peers send out disconnect packets
		m_ioService.poll();
	}
	
	// stop network (again; helpful to call before subsequent reset())
	m_ioService.stop();
	
	// reset network (allows reusing ioservice in future)
	m_ioService.reset();
	
	// finally, clear out peers (in case they're lingering)
	RecursiveGuard l(x_sessions);
	m_sessions.clear();
}

unsigned Host::protocolVersion() const
{
	return 3;
}

void Host::registerPeer(std::shared_ptr<Session> _s, CapDescs const& _caps)
{
	{
		clog(NetNote) << "p2p.host.peer.register" << _s->m_peer->id.abridged();
		RecursiveGuard l(x_sessions);
		// TODO: temporary loose-coupling; if m_peers already has peer,
		//       it is same as _s->m_peer. (fixing next PR)
		if (!m_peers.count(_s->m_peer->id))
			m_peers[_s->m_peer->id] = _s->m_peer;
		m_sessions[_s->m_peer->id] = _s;
	}
	
	unsigned o = (unsigned)UserPacket;
	for (auto const& i: _caps)
		if (haveCapability(i))
		{
			_s->m_capabilities[i] = shared_ptr<Capability>(m_capabilities[i]->newPeerCapability(_s.get(), o));
			o += m_capabilities[i]->messageCount();
		}
}

void Host::onNodeTableEvent(NodeId const& _n, NodeTableEventType const& _e)
{

	if (_e == NodeEntryAdded)
	{
		clog(NetNote) << "p2p.host.nodeTable.events.nodeEntryAdded " << _n;
		
		auto n = m_nodeTable->node(_n);
		if (n)
		{
			shared_ptr<Peer> p;
			{
				RecursiveGuard l(x_sessions);
				if (m_peers.count(_n))
					p = m_peers[_n];
				else
				{
					// TODO p2p: construct peer from node
					p.reset(new Peer());
					p->id = _n;
					p->endpoint = NodeIPEndpoint(n.endpoint.udp, n.endpoint.tcp);
					p->required = n.required;
					m_peers[_n] = p;
					
					clog(NetNote) << "p2p.host.peers.events.peersAdded " << _n << p->endpoint.tcp.address() << p->endpoint.udp.address();
				}
				p->endpoint.tcp = n.endpoint.tcp;
			}
			
			// TODO: Implement similar to discover. Attempt connecting to nodes
			//       until ideal peer count is reached; if all nodes are tried,
			//       repeat. Notably, this is an integrated process such that
			//       when onNodeTableEvent occurs we should also update +/-
			//       the list of nodes to be tried. Thus:
			//       1) externalize connection attempts
			//       2) attempt copies potentialPeer list
			//       3) replace this logic w/maintenance of potentialPeers
			if (peerCount() < m_idealPeerCount)
				connect(p);
		}
	}
	else if (_e == NodeEntryRemoved)
	{
		clog(NetNote) << "p2p.host.nodeTable.events.nodeEntryRemoved " << _n;
		
		RecursiveGuard l(x_sessions);
		m_peers.erase(_n);
	}
}

void Host::seal(bytes& _b)
{
	_b[0] = 0x22;
	_b[1] = 0x40;
	_b[2] = 0x08;
	_b[3] = 0x91;
	uint32_t len = (uint32_t)_b.size() - 8;
	_b[4] = (len >> 24) & 0xff;
	_b[5] = (len >> 16) & 0xff;
	_b[6] = (len >> 8) & 0xff;
	_b[7] = len & 0xff;
}

void Host::determinePublic(string const& _publicAddress, bool _upnp)
{
	m_peerAddresses.clear();
	
	// no point continuing if there are no interface addresses or valid listen port
	if (!m_ifAddresses.size() || m_listenPort < 1)
		return;

	// populate interfaces we'll listen on (eth listens on all interfaces); ignores local
	for (auto addr: m_ifAddresses)
		if ((m_netPrefs.localNetworking || !isPrivateAddress(addr)) && !isLocalHostAddress(addr))
			m_peerAddresses.insert(addr);
	
	// if user supplied address is a public address then we use it
	// if user supplied address is private, and localnetworking is enabled, we use it
	bi::address reqPublicAddr(bi::address(_publicAddress.empty() ? bi::address() : bi::address::from_string(_publicAddress)));
	bi::tcp::endpoint reqPublic(reqPublicAddr, m_listenPort);
	bool isprivate = isPrivateAddress(reqPublicAddr);
	bool ispublic = !isprivate && !isLocalHostAddress(reqPublicAddr);
	if (!reqPublicAddr.is_unspecified() && (ispublic || (isprivate && m_netPrefs.localNetworking)))
	{
		if (!m_peerAddresses.count(reqPublicAddr))
			m_peerAddresses.insert(reqPublicAddr);
		m_tcpPublic = reqPublic;
		return;
	}
	
	// if address wasn't provided, then use first public ipv4 address found
	for (auto addr: m_peerAddresses)
		if (addr.is_v4() && !isPrivateAddress(addr))
		{
			m_tcpPublic = bi::tcp::endpoint(*m_peerAddresses.begin(), m_listenPort);
			return;
		}
	
	// or find address via upnp
	if (_upnp)
	{
		bi::address upnpifaddr;
		bi::tcp::endpoint upnpep = Network::traverseNAT(m_ifAddresses, m_listenPort, upnpifaddr);
		if (!upnpep.address().is_unspecified() && !upnpifaddr.is_unspecified())
		{
			if (!m_peerAddresses.count(upnpep.address()))
				m_peerAddresses.insert(upnpep.address());
			m_tcpPublic = upnpep;
			return;
		}
	}

	// or if no address provided, use private ipv4 address if local networking is enabled
	if (reqPublicAddr.is_unspecified())
		if (m_netPrefs.localNetworking)
			for (auto addr: m_peerAddresses)
				if (addr.is_v4() && isPrivateAddress(addr))
				{
					m_tcpPublic = bi::tcp::endpoint(addr, m_listenPort);
					return;
				}
	
	// otherwise address is unspecified
	m_tcpPublic = bi::tcp::endpoint(bi::address(), m_listenPort);
}

void Host::runAcceptor()
{
	assert(m_listenPort > 0);
	
	if (m_run && !m_accepting)
	{
		clog(NetConnect) << "Listening on local port " << m_listenPort << " (public: " << m_tcpPublic << ")";
		m_accepting = true;

		// socket is created outside of acceptor-callback
		// An allocated socket is necessary as asio can use the socket
		// until the callback succeeds or fails.
		//
		// Until callback succeeds or fails, we can't dealloc it.
		//
		// Callback is guaranteed to be called via asio or when
		// m_tcp4Acceptor->stop() is called by Host.
		//
		// All exceptions are caught so they don't halt asio and so the
		// socket is deleted.
		//
		// It's possible for an accepted connection to return an error in which
		// case the socket may be open and must be closed to prevent asio from
		// processing socket events after socket is deallocated.
		
		bi::tcp::socket *s = new bi::tcp::socket(m_ioService);
		m_tcp4Acceptor.async_accept(*s, [=](boost::system::error_code ec)
		{
			// if no error code, doHandshake takes ownership
			bool success = false;
			if (!ec)
			{
				try
				{
					// doHandshake takes ownersihp of *s via std::move
					// incoming connection; we don't yet know nodeid
					auto handshake = make_shared<PeerHandshake>(m_alias, s);
					handshake->start();
					success = true;
				}
				catch (Exception const& _e)
				{
					clog(NetWarn) << "ERROR: " << diagnostic_information(_e);
				}
				catch (std::exception const& _e)
				{
					clog(NetWarn) << "ERROR: " << _e.what();
				}
			}
			
			// asio doesn't close socket on error
			if (!success)
			{
				if (s->is_open())
				{
					boost::system::error_code ec;
					s->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
					s->close();
				}
				delete s;
			}

			m_accepting = false;
			
			if (ec.value() < 1)
				runAcceptor();
		});
	}
}

void PeerHandshake::transition(boost::system::error_code _ech) {
	if (_ech)
	{
		boost::system::error_code ec;
		socket->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
		if (socket->is_open())
			socket->close();
		return;
	}
	
	auto self(shared_from_this());
	if (isNew())
	{
		clog(NetConnect) << "Authenticating connection for " << socket->remote_endpoint();
		
		if (originated)
		{
			clog(NetConnect) << "devp2p.connect.egress sending auth";
			// egress: tx auth
			asserts(remote);
			auth.resize(Signature::size + h256::size + Public::size + h256::size + 1);
			bytesConstRef sig(&auth[0], Signature::size);
			bytesConstRef hepubk(&auth[Signature::size], h256::size);
			bytesConstRef pubk(&auth[Signature::size + h256::size], Public::size);
			bytesConstRef nonce(&auth[Signature::size + h256::size + Public::size], h256::size);
			
			// E(remote-pubk, S(ecdhe-random, ecdh-shared-secret^nonce) || H(ecdhe-random-pubk) || pubk || nonce || 0x0)
			crypto::ecdh::agree(alias.sec(), remote, ss);
			sign(ecdhe.seckey(), ss ^ this->nonce).ref().copyTo(sig);
			sha3(ecdhe.pubkey().ref(), hepubk);
			alias.pub().ref().copyTo(pubk);
			this->nonce.ref().copyTo(nonce);
			auth[auth.size() - 1] = 0x0;
			encrypt(remote, &auth, authCipher);
			
			ba::async_write(*socket, ba::buffer(authCipher), [this, self](boost::system::error_code ec, std::size_t)
			{
				transition(ec);
			});
		}
		else
		{
			clog(NetConnect) << "devp2p.connect.ingress recving auth";
			// ingress: rx auth
			authCipher.resize(279);
			ba::async_read(*socket, ba::buffer(authCipher, 279), [this, self](boost::system::error_code ec, std::size_t)
			{
				if (ec)
					transition(ec);
				else
				{
					decrypt(alias.sec(), bytesConstRef(&authCipher), auth);
					bytesConstRef sig(&auth[0], Signature::size);
					bytesConstRef hepubk(&auth[Signature::size], h256::size);
					bytesConstRef pubk(&auth[Signature::size + h256::size], Public::size);
					bytesConstRef nonce(&auth[Signature::size + h256::size + Public::size], h256::size);
					pubk.copyTo(remote.ref());
					nonce.copyTo(remoteNonce.ref());
					
					crypto::ecdh::agree(alias.sec(), remote, ss);
					remoteEphemeral = recover(*(Signature*)sig.data(), ss ^ remoteNonce);
					assert(sha3(remoteEphemeral) == *(h256*)hepubk.data());
					transition();
				}
			});
		}
	}
	else if (isAcking())
		if (originated)
		{
			clog(NetConnect) << "devp2p.connect.egress recving ack";
			// egress: rx ack
			ackCipher.resize(182);
			ba::async_read(*socket, ba::buffer(ackCipher, 182), [this, self](boost::system::error_code ec, std::size_t)
			{
				if (ec)
					transition(ec);
				else
				{
					decrypt(alias.sec(), bytesConstRef(&ackCipher), ack);
					bytesConstRef(&ack).cropped(0, Public::size).copyTo(remoteEphemeral.ref());
					bytesConstRef(&ack).cropped(Public::size, h256::size).copyTo(remoteNonce.ref());
					transition();
				}
			});
		}
		else
		{
			clog(NetConnect) << "devp2p.connect.ingress sending ack";
			// ingress: tx ack
			ack.resize(Public::size + h256::size + 1);
			bytesConstRef epubk(&ack[0], Public::size);
			bytesConstRef nonce(&ack[Public::size], h256::size);
			ecdhe.pubkey().ref().copyTo(epubk);
			this->nonce.ref().copyTo(nonce);
			ack[ack.size() - 1] = 0x0;
			encrypt(remote, &ack, ackCipher);
			ba::async_write(*socket, ba::buffer(ackCipher), [this, self](boost::system::error_code ec, std::size_t)
			{
				transition(ec);
			});
		}
	else if (isAuthenticating())
	{
		if (originated)
			clog(NetConnect) << "devp2p.connect.egress sending magic sequence";
		else
			clog(NetConnect) << "devp2p.connect.ingress sending magic sequence";
		PeerSecrets* k = new PeerSecrets;
		bytes keyMaterialBytes(512);
		bytesConstRef keyMaterial(&keyMaterialBytes);

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
		
		// TESTING: send encrypt magic sequence
		bytes magic {0x22,0x40,0x08,0x91};
		encryptSymNoAuth(k->encryptK, &magic, k->magicCipherAndMac, h256());
		k->magicCipherAndMac.resize(k->magicCipherAndMac.size() + 32);
		sha3mac(k->egressMac.ref(), &magic, k->egressMac.ref());
		k->egressMac.ref().copyTo(bytesConstRef(&k->magicCipherAndMac).cropped(k->magicCipherAndMac.size() - 32, 32));
		
		clog(NetConnect) << "devp2p.connect.egress txrx magic sequence";
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
					clog(NetNote) << "devp2p.connect.egress recving magic sequence";
				else
					clog(NetNote) << "devp2p.connect.ingress recving magic sequence";
				
				if (ec)
				{
					delete k;
					transition(ec);
					return;
				}
				
				/// capabilities handshake (encrypted magic sequence is placeholder)
				bytes decryptedMagic;
				decryptSymNoAuth(k->encryptK, h256(), &k->recvdMagicCipherAndMac, decryptedMagic);
				if (decryptedMagic[0] == 0x22 && decryptedMagic[1] == 0x40 && decryptedMagic[2] == 0x08 && decryptedMagic[3] == 0x91)
				{
					shared_ptr<Peer> p;
					// todo: need host
//					p = m_peers[remote];
					
					if (!p)
					{
						p.reset(new Peer());
						p->id = remote;
					}
					p->endpoint.tcp.address(socket->remote_endpoint().address());
					p->m_lastDisconnect = NoDisconnect;
					p->m_lastConnected = std::chrono::system_clock::now();
					p->m_failedAttempts = 0;
					
					// todo: need host
//					auto ps = std::make_shared<Session>(this, move(*socket), p);
//					ps->start();
				}
				
				// todo: PeerSession needs to take ownership of k (PeerSecrets)
				delete k;
			});
		});
	}
	else
	{
		clog(NetConnect) << "Disconnecting " << socket->remote_endpoint() << " (Authentication Failed)";
		boost::system::error_code ec;
		socket->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
		socket->close();
	}
}

//void Host::doHandshake(PeerHandshake* _h, boost::system::error_code _ech)
//{
//	if (_ech)
//	{
//		boost::system::error_code ec;
//		_h->socket->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
//		if (_h->socket->is_open())
//			_h->socket->close();
//		delete _h;
//		return;
//	}
//	
//	if (_h->isNew())
//	{
//		clog(NetConnect) << "Authenticating connection for " << _h->socket->remote_endpoint();
//		
//		if (_h->originated)
//		{
//			clog(NetConnect) << "devp2p.connect.egress sending auth";
//			// egress: tx auth
//			asserts(_h->remote);
//			_h->auth.resize(Signature::size + h256::size + Public::size + h256::size + 1);
//			bytesConstRef sig(&_h->auth[0], Signature::size);
//			bytesConstRef hepubk(&_h->auth[Signature::size], h256::size);
//			bytesConstRef pubk(&_h->auth[Signature::size + h256::size], Public::size);
//			bytesConstRef nonce(&_h->auth[Signature::size + h256::size + Public::size], h256::size);
//			
//			// E(remote-pubk, S(ecdhe-random, ecdh-shared-secret^nonce) || H(ecdhe-random-pubk) || pubk || nonce || 0x0)
//			crypto::ecdh::agree(m_alias.sec(), _h->remote, _h->ss);
//			sign(_h->ecdhe.seckey(), _h->ss ^ _h->nonce).ref().copyTo(sig);
//			sha3(_h->ecdhe.pubkey().ref(), hepubk);
//			m_alias.pub().ref().copyTo(pubk);
//			_h->nonce.ref().copyTo(nonce);
//			_h->auth[_h->auth.size() - 1] = 0x0;
//			encrypt(_h->remote, &_h->auth, _h->authCipher);
//			ba::async_write(*_h->socket, ba::buffer(_h->authCipher), [=](boost::system::error_code ec, std::size_t)
//			{
//				doHandshake(_h, ec);
//			});
//		}
//		else
//		{
//			clog(NetConnect) << "devp2p.connect.ingress recving auth";
//			// ingress: rx auth
//			_h->authCipher.resize(279);
//			ba::async_read(*_h->socket, ba::buffer(_h->authCipher, 279), [=](boost::system::error_code ec, std::size_t)
//			{
//				if (ec)
//					doHandshake(_h, ec);
//				else
//				{
//					decrypt(m_alias.sec(), bytesConstRef(&_h->authCipher), _h->auth);
//					bytesConstRef sig(&_h->auth[0], Signature::size);
//					bytesConstRef hepubk(&_h->auth[Signature::size], h256::size);
//					bytesConstRef pubk(&_h->auth[Signature::size + h256::size], Public::size);
//					bytesConstRef nonce(&_h->auth[Signature::size + h256::size + Public::size], h256::size);
//					pubk.copyTo(_h->remote.ref());
//					nonce.copyTo(_h->remoteNonce.ref());
//					
//					crypto::ecdh::agree(m_alias.sec(), _h->remote, _h->ss);
//					_h->remoteEphemeral = recover(*(Signature*)sig.data(), _h->ss ^ _h->remoteNonce);
//					assert(sha3(_h->remoteEphemeral) == *(h256*)hepubk.data());
//					doHandshake(_h);
//				}
//			});
//		}
//	}
//	else if (_h->isAcking())
//		if (_h->originated)
//		{
//			clog(NetConnect) << "devp2p.connect.egress recving ack";
//			// egress: rx ack
//			_h->ackCipher.resize(182);
//			ba::async_read(*_h->socket, ba::buffer(_h->ackCipher, 182), [=](boost::system::error_code ec, std::size_t)
//			{
//				if (ec)
//					doHandshake(_h, ec);
//				else
//				{
//					decrypt(m_alias.sec(), bytesConstRef(&_h->ackCipher), _h->ack);
//					bytesConstRef(&_h->ack).cropped(0, Public::size).copyTo(_h->remoteEphemeral.ref());
//					bytesConstRef(&_h->ack).cropped(Public::size, h256::size).copyTo(_h->remoteNonce.ref());
//					doHandshake(_h);
//				}
//			});
//		}
//		else
//		{
//			clog(NetConnect) << "devp2p.connect.ingress sending ack";
//			// ingress: tx ack
//			_h->ack.resize(Public::size + h256::size + 1);
//			bytesConstRef epubk(&_h->ack[0], Public::size);
//			bytesConstRef nonce(&_h->ack[Public::size], h256::size);
//			_h->ecdhe.pubkey().ref().copyTo(epubk);
//			_h->nonce.ref().copyTo(nonce);
//			_h->ack[_h->ack.size() - 1] = 0x0;
//			encrypt(_h->remote, &_h->ack, _h->ackCipher);
//			ba::async_write(*_h->socket, ba::buffer(_h->ackCipher), [=](boost::system::error_code ec, std::size_t)
//			{
//				doHandshake(_h, ec);
//			});
//		}
//	else if (_h->isAuthenticating())
//	{
//		if (_h->originated)
//			clog(NetConnect) << "devp2p.connect.egress sending magic sequence";
//		else
//			clog(NetConnect) << "devp2p.connect.ingress sending magic sequence";
//		PeerSecrets* k = new PeerSecrets;
//		bytes keyMaterialBytes(512);
//		bytesConstRef keyMaterial(&keyMaterialBytes);
//
//		_h->ecdhe.agree(_h->remoteEphemeral, _h->ess);
//		_h->ess.ref().copyTo(keyMaterial.cropped(0, h256::size));
//		_h->ss.ref().copyTo(keyMaterial.cropped(h256::size, h256::size));
//		//		auto token = sha3(ssA);
//		k->encryptK = sha3(keyMaterial);
//		k->encryptK.ref().copyTo(keyMaterial.cropped(h256::size, h256::size));
//		k->macK = sha3(keyMaterial);
//		
//		// Initiator egress-mac: sha3(mac-secret^recipient-nonce || auth-sent-init)
//		//           ingress-mac: sha3(mac-secret^initiator-nonce || auth-recvd-ack)
//		// Recipient egress-mac: sha3(mac-secret^initiator-nonce || auth-sent-ack)
//		//           ingress-mac: sha3(mac-secret^recipient-nonce || auth-recvd-init)
//		
//		bytes const& egressCipher = _h->originated ? _h->authCipher : _h->ackCipher;
//		keyMaterialBytes.resize(h256::size + egressCipher.size());
//		keyMaterial.retarget(keyMaterialBytes.data(), keyMaterialBytes.size());
//		(k->macK ^ _h->remoteNonce).ref().copyTo(keyMaterial);
//		bytesConstRef(&egressCipher).copyTo(keyMaterial.cropped(h256::size, egressCipher.size()));
//		k->egressMac = sha3(keyMaterial);
//		
//		bytes const& ingressCipher = _h->originated ? _h->ackCipher : _h->authCipher;
//		keyMaterialBytes.resize(h256::size + ingressCipher.size());
//		keyMaterial.retarget(keyMaterialBytes.data(), keyMaterialBytes.size());
//		(k->macK ^ _h->nonce).ref().copyTo(keyMaterial);
//		bytesConstRef(&ingressCipher).copyTo(keyMaterial.cropped(h256::size, ingressCipher.size()));
//		k->ingressMac = sha3(keyMaterial);
//		
//		// TESTING: send encrypt magic sequence
//		bytes magic {0x22,0x40,0x08,0x91};
//		encryptSymNoAuth(k->encryptK, &magic, k->magicCipherAndMac, h256());
//		k->magicCipherAndMac.resize(k->magicCipherAndMac.size() + 32);
//		sha3mac(k->egressMac.ref(), &magic, k->egressMac.ref());
//		k->egressMac.ref().copyTo(bytesConstRef(&k->magicCipherAndMac).cropped(k->magicCipherAndMac.size() - 32, 32));
//		
//		clog(NetConnect) << "devp2p.connect.egress txrx magic sequence";
//		k->recvdMagicCipherAndMac.resize(k->magicCipherAndMac.size());
//		
//		ba::async_write(*_h->socket, ba::buffer(k->magicCipherAndMac), [this, k, _h, magic](boost::system::error_code ec, std::size_t)
//		{
//			if (ec)
//			{
//				delete k;
//				doHandshake(_h, ec);
//				return;
//			}
//			
//			ba::async_read(*_h->socket, ba::buffer(k->recvdMagicCipherAndMac, k->magicCipherAndMac.size()), [this, k, _h, magic](boost::system::error_code ec, std::size_t)
//			{
//				if (_h->originated)
//					clog(NetNote) << "devp2p.connect.egress recving magic sequence";
//				else
//					clog(NetNote) << "devp2p.connect.ingress recving magic sequence";
//				
//				if (ec)
//				{
//					delete k;
//					doHandshake(_h, ec);
//					return;
//				}
//				
//				/// capabilities handshake (encrypted magic sequence is placeholder)
//				bytes decryptedMagic;
//				decryptSymNoAuth(k->encryptK, h256(), &k->recvdMagicCipherAndMac, decryptedMagic);
//				if (decryptedMagic[0] == 0x22 && decryptedMagic[1] == 0x40 && decryptedMagic[2] == 0x08 && decryptedMagic[3] == 0x91)
//				{
//					shared_ptr<Peer> p;
//					p = m_peers[_h->remote];
//					
//					if (!p)
//					{
//						p.reset(new Peer());
//						p->id = _h->remote;
//					}
//					p->endpoint.tcp.address(_h->socket->remote_endpoint().address());
//					p->m_lastDisconnect = NoDisconnect;
//					p->m_lastConnected = std::chrono::system_clock::now();
//					p->m_failedAttempts = 0;
//					
//					auto ps = std::make_shared<Session>(this, move(*_h->socket), p);
//					ps->start();
//				}
//				
//				delete k;
//			});
//		});
//	}
//	else
//	{
//		clog(NetConnect) << "Disconnecting " << _h->socket->remote_endpoint() << " (Authentication Failed)";
//		boost::system::error_code ec;
//		_h->socket->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
//		_h->socket->close();
//		delete _h;
//	}
//}

string Host::pocHost()
{
	vector<string> strs;
	boost::split(strs, dev::Version, boost::is_any_of("."));
	return "poc-" + strs[1] + ".ethdev.com";
}

void Host::addNode(NodeId const& _node, std::string const& _addr, unsigned short _tcpPeerPort, unsigned short _udpNodePort)
{
	// TODO: p2p clean this up (bring tested acceptor code over from network branch)
	while (isWorking() && !m_run)
		this_thread::sleep_for(chrono::milliseconds(50));
	if (!m_run)
		return;
	
	if (_tcpPeerPort < 30300 || _tcpPeerPort > 30305)
		cwarn << "Non-standard port being recorded: " << _tcpPeerPort;
	
	if (_tcpPeerPort >= /*49152*/32768)
	{
		cwarn << "Private port being recorded - setting to 0";
		_tcpPeerPort = 0;
	}
	
	boost::system::error_code ec;
	bi::address addr = bi::address::from_string(_addr, ec);
	if (ec)
	{
		bi::tcp::resolver *r = new bi::tcp::resolver(m_ioService);
		r->async_resolve({_addr, toString(_tcpPeerPort)}, [=](boost::system::error_code const& _ec, bi::tcp::resolver::iterator _epIt)
		{
			if (!_ec)
			{
				bi::tcp::endpoint tcp = *_epIt;
				if (m_nodeTable) m_nodeTable->addNode(Node(_node, NodeIPEndpoint(bi::udp::endpoint(tcp.address(), _udpNodePort), tcp)));
			}
			delete r;
		});
	}
	else
		if (m_nodeTable) m_nodeTable->addNode(Node(_node, NodeIPEndpoint(bi::udp::endpoint(addr, _udpNodePort), bi::tcp::endpoint(addr, _tcpPeerPort))));
}

void Host::connect(std::shared_ptr<Peer> const& _p)
{
	for (unsigned i = 0; i < 200; i++)
		if (isWorking() && !m_run)
			this_thread::sleep_for(chrono::milliseconds(50));
	if (!m_run)
		return;
	
	if (havePeerSession(_p->id))
	{
		clog(NetWarn) << "Aborted connect. Node already connected.";
		return;
	}
	
	if (!m_nodeTable->haveNode(_p->id))
	{
		clog(NetWarn) << "Aborted connect. Node not in node table.";
		m_nodeTable->addNode(*_p.get());
		return;
	}
	
	// prevent concurrently connecting to a node
	Peer *nptr = _p.get();
	{
		Guard l(x_pendingNodeConns);
		if (m_pendingPeerConns.count(nptr))
			return;
		m_pendingPeerConns.insert(nptr);
	}
	
	clog(NetConnect) << "Attempting connection to node" << _p->id.abridged() << "@" << _p->peerEndpoint() << "from" << id().abridged();
	bi::tcp::socket* s = new bi::tcp::socket(m_ioService);
	s->async_connect(_p->peerEndpoint(), [=](boost::system::error_code const& ec)
	{
		if (ec)
		{
			clog(NetConnect) << "Connection refused to node" << _p->id.abridged() << "@" << _p->peerEndpoint() << "(" << ec.message() << ")";
			_p->m_lastDisconnect = TCPError;
			_p->m_lastAttempted = std::chrono::system_clock::now();
			delete s;
		}
		else
		{
			clog(NetConnect) << "Connected to" << _p->id.abridged() << "@" << _p->peerEndpoint();
			auto handshake = make_shared<PeerHandshake>(m_alias, s, _p->id);
			handshake->start();
		}
		Guard l(x_pendingNodeConns);
		m_pendingPeerConns.erase(nptr);
	});
}

PeerSessionInfos Host::peerSessionInfo() const
{
	if (!m_run)
		return PeerSessionInfos();

	std::vector<PeerSessionInfo> ret;
	RecursiveGuard l(x_sessions);
	for (auto& i: m_sessions)
		if (auto j = i.second.lock())
			if (j->isConnected())
				ret.push_back(j->m_info);
	return ret;
}

size_t Host::peerCount() const
{
	unsigned retCount = 0;
	RecursiveGuard l(x_sessions);
	for (auto& i: m_sessions)
		if (std::shared_ptr<Session> j = i.second.lock())
			if (j->isConnected())
				retCount++;
	return retCount;
}

void Host::run(boost::system::error_code const&)
{
	if (!m_run)
	{
		// reset NodeTable
		m_nodeTable.reset();
		
		// stopping io service allows running manual network operations for shutdown
		// and also stops blocking worker thread, allowing worker thread to exit
		m_ioService.stop();
		
		// resetting timer signals network that nothing else can be scheduled to run
		m_timer.reset();
		return;
	}
	
	m_nodeTable->processEvents();
	
	for (auto p: m_sessions)
		if (auto pp = p.second.lock())
			pp->serviceNodesRequest();
	
//	keepAlivePeers();
//	disconnectLatePeers();

	auto c = peerCount();
	if (m_idealPeerCount && !c)
		for (auto p: m_peers)
			if (p.second->shouldReconnect())
			{
				// TODO p2p: fixme
				p.second->m_lastAttempted = std::chrono::system_clock::now();
				connect(p.second);
				break;
			}
	
	if (c < m_idealPeerCount)
		m_nodeTable->discover();
	
	auto runcb = [this](boost::system::error_code const& error) { run(error); };
	m_timer->expires_from_now(boost::posix_time::milliseconds(c_timerInterval));
	m_timer->async_wait(runcb);
}
			
void Host::startedWorking()
{
	asserts(!m_timer);

	{
		// prevent m_run from being set to true at same time as set to false by stop()
		// don't release mutex until m_timer is set so in case stop() is called at same
		// time, stop will wait on m_timer and graceful network shutdown.
		Guard l(x_runTimer);
		// create deadline timer
		m_timer.reset(new boost::asio::deadline_timer(m_ioService));
		m_run = true;
	}
	
	// try to open acceptor (todo: ipv6)
	m_listenPort = Network::tcp4Listen(m_tcp4Acceptor, m_netPrefs.listenPort);
	
	// start capability threads
	for (auto const& h: m_capabilities)
		h.second->onStarting();
	
	// determine public IP, but only if we're able to listen for connections
	// todo: GUI when listen is unavailable in UI
	if (m_listenPort)
	{
		determinePublic(m_netPrefs.publicIP, m_netPrefs.upnp);
		
		if (m_listenPort > 0)
			runAcceptor();
	}
	else
		clog(NetNote) << "p2p.start.notice id:" << id().abridged() << "Listen port is invalid or unavailable. Node Table using default port (30303).";
	
	// TODO: add m_tcpPublic endpoint; sort out endpoint stuff for nodetable
	m_nodeTable.reset(new NodeTable(m_ioService, m_alias, m_listenPort > 0 ? m_listenPort : 30303));
	m_nodeTable->setEventHandler(new HostNodeTableHandler(*this));
	restoreNetwork(&m_restoreNetwork);
	
	clog(NetNote) << "p2p.started id:" << id().abridged();
	
	run(boost::system::error_code());
}

void Host::doWork()
{
	if (m_run)
		m_ioService.run();
}

void Host::keepAlivePeers()
{
	if (chrono::steady_clock::now() - c_keepAliveInterval < m_lastPing)
		return;
	
	RecursiveGuard l(x_sessions);
	for (auto p: m_sessions)
		if (auto pp = p.second.lock())
				pp->ping();

	m_lastPing = chrono::steady_clock::now();
}

void Host::disconnectLatePeers()
{
	auto now = chrono::steady_clock::now();
	if (now - c_keepAliveTimeOut < m_lastPing)
		return;

	RecursiveGuard l(x_sessions);
	for (auto p: m_sessions)
		if (auto pp = p.second.lock())
			if (now - c_keepAliveTimeOut > m_lastPing && pp->m_lastReceived < m_lastPing)
				pp->disconnect(PingTimeout);
}

bytes Host::saveNetwork() const
{
	std::list<Peer> peers;
	{
		RecursiveGuard l(x_sessions);
		for (auto p: m_peers)
			if (p.second)
				peers.push_back(*p.second);
	}
	peers.sort();
	
	RLPStream network;
	int count = 0;
	{
		RecursiveGuard l(x_sessions);
		for (auto const& p: peers)
		{
			// TODO: alpha: Figure out why it ever shares these ports.//p.address.port() >= 30300 && p.address.port() <= 30305 &&
			// TODO: alpha: if/how to save private addresses
			// Only save peers which have connected within 2 days, with properly-advertised port and public IP address
			if (chrono::system_clock::now() - p.m_lastConnected < chrono::seconds(3600 * 48) && p.peerEndpoint().port() > 0 && p.peerEndpoint().port() < /*49152*/32768 && p.id != id() && !isPrivateAddress(p.peerEndpoint().address()))
			{
				network.appendList(10);
				if (p.peerEndpoint().address().is_v4())
					network << p.peerEndpoint().address().to_v4().to_bytes();
				else
					network << p.peerEndpoint().address().to_v6().to_bytes();
				// TODO: alpha: replace 0 with trust-state of node
				network << p.peerEndpoint().port() << p.id << 0
					<< chrono::duration_cast<chrono::seconds>(p.m_lastConnected.time_since_epoch()).count()
					<< chrono::duration_cast<chrono::seconds>(p.m_lastAttempted.time_since_epoch()).count()
					<< p.m_failedAttempts << (unsigned)p.m_lastDisconnect << p.m_score << p.m_rating;
				count++;
			}
		}
	}

	if (!!m_nodeTable)
	{
		auto state = m_nodeTable->snapshot();
		state.sort();
		for (auto const& s: state)
		{
			network.appendList(3);
			if (s.endpoint.tcp.address().is_v4())
				network << s.endpoint.tcp.address().to_v4().to_bytes();
			else
				network << s.endpoint.tcp.address().to_v6().to_bytes();
			network << s.endpoint.tcp.port() << s.id;
			count++;
		}
	}
	
	RLPStream ret(3);
	ret << 1 << m_alias.secret();
	ret.appendList(count).appendRaw(network.out(), count);
	return ret.out();
}

void Host::restoreNetwork(bytesConstRef _b)
{
	// nodes can only be added if network is added
	if (!isStarted())
		BOOST_THROW_EXCEPTION(NetworkStartRequired());
	
	RecursiveGuard l(x_sessions);
	RLP r(_b);
	if (r.itemCount() > 0 && r[0].isInt() && r[0].toInt<int>() == 1)
	{
		// r[0] = version
		// r[1] = key
		// r[2] = nodes

		for (auto i: r[2])
		{
			bi::tcp::endpoint tcp;
			bi::udp::endpoint udp;
			if (i[0].itemCount() == 4)
			{
				tcp = bi::tcp::endpoint(bi::address_v4(i[0].toArray<byte, 4>()), i[1].toInt<short>());
				udp = bi::udp::endpoint(bi::address_v4(i[0].toArray<byte, 4>()), i[1].toInt<short>());
			}
			else
			{
				tcp = bi::tcp::endpoint(bi::address_v6(i[0].toArray<byte, 16>()), i[1].toInt<short>());
				udp = bi::udp::endpoint(bi::address_v6(i[0].toArray<byte, 16>()), i[1].toInt<short>());
			}
			auto id = (NodeId)i[2];
			if (i.itemCount() == 3)
				m_nodeTable->addNode(id, udp, tcp);
			else if (i.itemCount() == 10)
			{
				shared_ptr<Peer> p = make_shared<Peer>();
				p->id = id;
				p->m_lastConnected = chrono::system_clock::time_point(chrono::seconds(i[4].toInt<unsigned>()));
				p->m_lastAttempted = chrono::system_clock::time_point(chrono::seconds(i[5].toInt<unsigned>()));
				p->m_failedAttempts = i[6].toInt<unsigned>();
				p->m_lastDisconnect = (DisconnectReason)i[7].toInt<unsigned>();
				p->m_score = (int)i[8].toInt<unsigned>();
				p->m_rating = (int)i[9].toInt<unsigned>();
				p->endpoint.tcp = tcp;
				p->endpoint.udp = udp;
				m_peers[p->id] = p;
				m_nodeTable->addNode(*p.get());
			}
		}
	}
}

KeyPair Host::networkAlias(bytesConstRef _b)
{
	RLP r(_b);
	if (r.itemCount() == 3 && r[0].isInt() && r[0].toInt<int>() == 1)
		return move(KeyPair(move(Secret(r[1].toBytes()))));
	else
		return move(KeyPair::create());
}
