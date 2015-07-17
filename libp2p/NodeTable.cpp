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
/** @file NodeTable.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 */

#include "NodeTable.h"
using namespace std;
using namespace dev;
using namespace dev::p2p;

const char* NodeTableWarn::name() { return "!P!"; }
const char* NodeTableNote::name() { return "*P*"; }
const char* NodeTableMessageSummary::name() { return "-P-"; }
const char* NodeTableMessageDetail::name() { return "=P="; }
const char* NodeTableConnect::name() { return "+P+"; }
const char* NodeTableEvent::name() { return "+P+"; }
const char* NodeTableTimer::name() { return "+P+"; }
const char* NodeTableUpdate::name() { return "+P+"; }
const char* NodeTableTriviaSummary::name() { return "-P-"; }
const char* NodeTableTriviaDetail::name() { return "=P="; }
const char* NodeTableAllDetail::name() { return "=P="; }
const char* NodeTableEgress::name() { return ">>P"; }
const char* NodeTableIngress::name() { return "<<P"; }

NodeEntry::NodeEntry(NodeId const& _src, Public const& _pubk, NodeIPEndpoint const& _gw): Node(_pubk, _gw), distance(NodeTable::distance(_src, _pubk)) {}

NodeTable::NodeTable(ba::io_service& _io, KeyPair const& _alias, NodeIPEndpoint const& _endpoint, bool _enabled):
	m_node(Node(_alias.pub(), _endpoint)),
	m_secret(_alias.sec()),
	m_socket(new NodeSocket(_io, *this, (bi::udp::endpoint)m_node.endpoint)),
	m_socketPointer(m_socket.get()),
	m_timers(_io)
{
	for (unsigned i = 0; i < s_bins; i++)
		m_state[i].distance = i;
	
	if (!_enabled)
		return;
	
	try
	{
		m_socketPointer->connect();
		doDiscovery();
	}
	catch (std::exception const& _e)
	{
		clog(NetWarn) << "Exception connecting NodeTable socket: " << _e.what();
		clog(NetWarn) << "Discovery disabled.";
	}
}
	
NodeTable::~NodeTable()
{
	m_timers.stop();
	m_socketPointer->disconnect();
}

void NodeTable::processEvents()
{
	if (m_nodeEventHandler)
		m_nodeEventHandler->processEvents();
}

shared_ptr<NodeEntry> NodeTable::addNode(Node const& _node, NodeRelation _relation)
{
	if (_relation == Known)
	{
		shared_ptr<NodeEntry> ret(new NodeEntry(m_node.id, _node.id, _node.endpoint));
		ret->pending = false;
		DEV_GUARDED(x_nodes)
			m_nodes[_node.id] = ret;
		noteActiveNode(_node.id, _node.endpoint);
		return ret;
	}
	
	if (!_node.endpoint)
		return shared_ptr<NodeEntry>();
	
	// ping address to recover nodeid if nodeid is empty
	if (!_node.id)
	{
		DEV_GUARDED(x_nodes)
			clog(NodeTableConnect) << "Sending public key discovery Ping to" << (bi::udp::endpoint)_node.endpoint << "(Advertising:" << (bi::udp::endpoint)m_node.endpoint << ")";
		DEV_GUARDED(x_pubkDiscoverPings)
			m_pubkDiscoverPings[_node.endpoint.address] = std::chrono::steady_clock::now();
		ping(_node.endpoint);
		return shared_ptr<NodeEntry>();
	}
	
	DEV_GUARDED(x_nodes)
		if (m_nodes.count(_node.id))
			return m_nodes[_node.id];
	
	shared_ptr<NodeEntry> ret(new NodeEntry(m_node.id, _node.id, _node.endpoint));
	DEV_GUARDED(x_nodes)
		m_nodes[_node.id] = ret;
	clog(NodeTableConnect) << "addNode pending for" << _node.endpoint;
	ping(_node.endpoint);
	return ret;
}

list<NodeId> NodeTable::nodes() const
{
	list<NodeId> nodes;
	DEV_GUARDED(x_nodes)
		for (auto& i: m_nodes)
			nodes.push_back(i.second->id);
	return nodes;
}

list<NodeEntry> NodeTable::snapshot() const
{
	list<NodeEntry> ret;
	DEV_GUARDED(x_state)
		for (auto const& s: m_state)
			for (auto const& np: s.nodes)
				if (auto n = np.lock())
					ret.push_back(*n);
	return ret;
}

Node NodeTable::node(NodeId const& _id)
{
	Guard l(x_nodes);
	if (m_nodes.count(_id))
	{
		auto entry = m_nodes[_id];
		return Node(_id, entry->endpoint, entry->required);
	}
	return UnspecifiedNode;
}

shared_ptr<NodeEntry> NodeTable::nodeEntry(NodeId _id)
{
	Guard l(x_nodes);
	return m_nodes.count(_id) ? m_nodes[_id] : shared_ptr<NodeEntry>();
}

void NodeTable::doDiscover(NodeId _node, unsigned _round, shared_ptr<set<shared_ptr<NodeEntry>>> _tried)
{
	// NOTE: ONLY called by doDiscovery!
	
	if (!m_socketPointer->isOpen())
		return;
	
	if (_round == s_maxSteps)
	{
		clog(NodeTableEvent) << "Terminating discover after " << _round << " rounds.";
		doDiscovery();
		return;
	}
	else if (!_round && !_tried)
		// initialized _tried on first round
		_tried.reset(new set<shared_ptr<NodeEntry>>());
	
	auto nearest = nearestNodeEntries(_node);
	list<shared_ptr<NodeEntry>> tried;
	for (unsigned i = 0; i < nearest.size() && tried.size() < s_alpha; i++)
		if (!_tried->count(nearest[i]))
		{
			auto r = nearest[i];
			tried.push_back(r);
			FindNode p(r->endpoint, _node);
			p.sign(m_secret);
			DEV_GUARDED(x_findNodeTimeout)
				m_findNodeTimeout.push_back(make_pair(r->id, chrono::steady_clock::now()));
			m_socketPointer->send(p);
		}
	
	if (tried.empty())
	{
		clog(NodeTableEvent) << "Terminating discover after " << _round << " rounds.";
		doDiscovery();
		return;
	}
		
	while (!tried.empty())
	{
		_tried->insert(tried.front());
		tried.pop_front();
	}

	m_timers.schedule(c_reqTimeout.count() * 2, [this, _node, _round, _tried](boost::system::error_code const& _ec)
	{
		if (_ec)
			clog(NodeTableWarn) << "Discovery timer canceled!";
		doDiscover(_node, _round + 1, _tried);
	});
}

vector<shared_ptr<NodeEntry>> NodeTable::nearestNodeEntries(NodeId _target)
{
	// send s_alpha FindNode packets to nodes we know, closest to target
	static unsigned lastBin = s_bins - 1;
	unsigned head = distance(m_node.id, _target);
	unsigned tail = head == 0 ? lastBin : (head - 1) % s_bins;
	
	map<unsigned, list<shared_ptr<NodeEntry>>> found;
	unsigned count = 0;
	
	// if d is 0, then we roll look forward, if last, we reverse, else, spread from d
	if (head > 1 && tail != lastBin)
		while (head != tail && head < s_bins && count < s_bucketSize)
		{
			Guard l(x_state);
			for (auto const& n: m_state[head].nodes)
				if (auto p = n.lock())
				{
					if (count < s_bucketSize)
						found[distance(_target, p->id)].push_back(p);
					else
						break;
				}
			
			if (count < s_bucketSize && tail)
				for (auto const& n: m_state[tail].nodes)
					if (auto p = n.lock())
					{
						if (count < s_bucketSize)
							found[distance(_target, p->id)].push_back(p);
						else
							break;
					}

			head++;
			if (tail)
				tail--;
		}
	else if (head < 2)
		while (head < s_bins && count < s_bucketSize)
		{
			Guard l(x_state);
			for (auto const& n: m_state[head].nodes)
				if (auto p = n.lock())
				{
					if (count < s_bucketSize)
						found[distance(_target, p->id)].push_back(p);
					else
						break;
				}
			head++;
		}
	else
		while (tail > 0 && count < s_bucketSize)
		{
			Guard l(x_state);
			for (auto const& n: m_state[tail].nodes)
				if (auto p = n.lock())
				{
					if (count < s_bucketSize)
						found[distance(_target, p->id)].push_back(p);
					else
						break;
				}
			tail--;
		}
	
	vector<shared_ptr<NodeEntry>> ret;
	for (auto& nodes: found)
		for (auto const& n: nodes.second)
			if (ret.size() < s_bucketSize && !!n->endpoint && n->endpoint.isAllowed())
				ret.push_back(n);
	return ret;
}

void NodeTable::ping(NodeIPEndpoint _to) const
{
	NodeIPEndpoint src;
	DEV_GUARDED(x_nodes)
		src = m_node.endpoint;
	PingNode p(src, _to);
	p.sign(m_secret);
	m_socketPointer->send(p);
}

void NodeTable::ping(NodeEntry* _n) const
{
	if (_n)
		ping(_n->endpoint);
}

void NodeTable::evict(shared_ptr<NodeEntry> _leastSeen, shared_ptr<NodeEntry> _new)
{
	if (!m_socketPointer->isOpen())
		return;
	
	unsigned evicts;
	DEV_GUARDED(x_evictions)
	{
		m_evictions.push_back(EvictionTimeout(make_pair(_leastSeen->id,chrono::steady_clock::now()), _new->id));
		evicts = m_evictions.size();
	}

	if (evicts == 1)
		doCheckEvictions();
	ping(_leastSeen.get());
}

void NodeTable::noteActiveNode(Public const& _pubk, bi::udp::endpoint const& _endpoint)
{
	if (_pubk == m_node.address() || !NodeIPEndpoint(_endpoint.address(), _endpoint.port(), _endpoint.port()).isAllowed())
		return;

	shared_ptr<NodeEntry> node = nodeEntry(_pubk);
	if (!!node && !node->pending)
	{
		clog(NodeTableConnect) << "Noting active node:" << _pubk << _endpoint.address().to_string() << ":" << _endpoint.port();
		node->endpoint.address = _endpoint.address();
		node->endpoint.udpPort = _endpoint.port();
		
		shared_ptr<NodeEntry> contested;
		{
			Guard l(x_state);
			NodeBucket& s = bucket_UNSAFE(node.get());
			bool removed = false;
			s.nodes.remove_if([&node, &removed](weak_ptr<NodeEntry> const& n)
			{
				if (n.lock() == node)
					removed = true;
				return removed;
			});
			
			if (s.nodes.size() >= s_bucketSize)
			{
				if (removed)
					clog(NodeTableWarn) << "DANGER: Bucket overflow when swapping node position.";
				
				// It's only contested iff nodeentry exists
				contested = s.nodes.front().lock();
				if (!contested)
				{
					s.nodes.pop_front();
					s.nodes.push_back(node);
					if (!removed && m_nodeEventHandler)
						m_nodeEventHandler->appendEvent(node->id, NodeEntryAdded);
				}
			}
			else
			{
				s.nodes.push_back(node);
				if (!removed && m_nodeEventHandler)
					m_nodeEventHandler->appendEvent(node->id, NodeEntryAdded);
			}
		}
		
		if (contested)
			evict(contested, node);
	}
}

void NodeTable::dropNode(shared_ptr<NodeEntry> _n)
{
	// remove from nodetable
	{
		Guard l(x_state);
		NodeBucket& s = bucket_UNSAFE(_n.get());
		s.nodes.remove_if([&_n](weak_ptr<NodeEntry> n) { return n.lock() == _n; });
	}
	
	// notify host
	clog(NodeTableUpdate) << "p2p.nodes.drop " << _n->id;
	if (m_nodeEventHandler)
		m_nodeEventHandler->appendEvent(_n->id, NodeEntryDropped);
}

NodeTable::NodeBucket& NodeTable::bucket_UNSAFE(NodeEntry const* _n)
{
	return m_state[_n->distance - 1];
}

void NodeTable::onReceived(UDPSocketFace*, bi::udp::endpoint const& _from, bytesConstRef _packet)
{
	// h256 + Signature + type + RLP (smallest possible packet is empty neighbours packet which is 3 bytes)
	if (_packet.size() < h256::size + Signature::size + 1 + 3)
	{
		clog(NodeTableTriviaSummary) << "Invalid message size from " << _from.address().to_string() << ":" << _from.port();
		return;
	}
	
	bytesConstRef hashedBytes(_packet.cropped(h256::size, _packet.size() - h256::size));
	h256 hashSigned(sha3(hashedBytes));
	if (!_packet.cropped(0, h256::size).contentsEqual(hashSigned.asBytes()))
	{
		clog(NodeTableTriviaSummary) << "Invalid message hash from " << _from.address().to_string() << ":" << _from.port();
		return;
	}
	
	bytesConstRef signedBytes(hashedBytes.cropped(Signature::size, hashedBytes.size() - Signature::size));

	// todo: verify sig via known-nodeid and MDC
	
	bytesConstRef sigBytes(_packet.cropped(h256::size, Signature::size));
	Public nodeid(dev::recover(*(Signature const*)sigBytes.data(), sha3(signedBytes)));
	if (!nodeid)
	{
		clog(NodeTableTriviaSummary) << "Invalid message signature from " << _from.address().to_string() << ":" << _from.port();
		return;
	}
	
	unsigned packetType = signedBytes[0];
	bytesConstRef rlpBytes(_packet.cropped(h256::size + Signature::size + 1));
	try {
		RLP rlp(rlpBytes);
		switch (packetType)
		{
			case Pong::type:
			{
				Pong in = Pong::fromBytesConstRef(_from, rlpBytes);
				
				// whenever a pong is received, check if it's in m_evictions
				bool found = false;
				EvictionTimeout evictionEntry;
				DEV_GUARDED(x_evictions)
					for (auto it = m_evictions.begin(); it != m_evictions.end(); ++it)
						if (it->first.first == nodeid && it->first.second > std::chrono::steady_clock::now())
						{
							found = true;
							evictionEntry = *it;
							m_evictions.erase(it);
							break;
						}
				if (found)
				{
					if (auto n = nodeEntry(evictionEntry.second))
						dropNode(n);
					if (auto n = nodeEntry(evictionEntry.first.first))
						n->pending = false;
				}
				else
				{
					// if not, check if it's known/pending or a pubk discovery ping
					if (auto n = nodeEntry(nodeid))
						n->pending = false;
					else
					{
						DEV_GUARDED(x_pubkDiscoverPings)
						{
							if (!m_pubkDiscoverPings.count(_from.address()))
								return; // unsolicited pong; don't note node as active
							m_pubkDiscoverPings.erase(_from.address());
						}
						if (!haveNode(nodeid))
							addNode(Node(nodeid, NodeIPEndpoint(_from.address(), _from.port(), _from.port())));
					}
				}
				
				// update our endpoint address and UDP port
				DEV_GUARDED(x_nodes)
				{
					if ((!m_node.endpoint || !m_node.endpoint.isAllowed()) && isPublicAddress(in.destination.address))
						m_node.endpoint.address = in.destination.address;
					m_node.endpoint.udpPort = in.destination.udpPort;
				}
				
				clog(NodeTableConnect) << "PONG from " << nodeid << _from;
				break;
			}
				
			case Neighbours::type:
			{
				bool expected = false;
				auto now = chrono::steady_clock::now();
				DEV_GUARDED(x_findNodeTimeout)
					m_findNodeTimeout.remove_if([&](NodeIdTimePoint const& t)
					{
						if (t.first == nodeid && now - t.second < c_reqTimeout)
							expected = true;
						else if (t.first == nodeid)
							return true;
						return false;
					});
				
				if (!expected)
				{
					clog(NetConnect) << "Dropping unsolicited neighbours packet from " << _from.address();
					break;
				}
				
				Neighbours in = Neighbours::fromBytesConstRef(_from, rlpBytes);
				for (auto n: in.neighbours)
					addNode(Node(n.node, n.endpoint));
				break;
			}

			case FindNode::type:
			{
				FindNode in = FindNode::fromBytesConstRef(_from, rlpBytes);
				if (RLPXDatagramFace::secondsSinceEpoch() > in.ts)
				{
					clog(NodeTableTriviaSummary) << "Received expired FindNode from " << _from.address().to_string() << ":" << _from.port();
					return;
				}

				vector<shared_ptr<NodeEntry>> nearest = nearestNodeEntries(in.target);
				static unsigned const nlimit = (m_socketPointer->maxDatagramSize - 109) / 90;
				for (unsigned offset = 0; offset < nearest.size(); offset += nlimit)
				{
					Neighbours out(_from, nearest, offset, nlimit);
					out.sign(m_secret);
					if (out.data.size() > 1280)
						clog(NetWarn) << "Sending truncated datagram, size: " << out.data.size();
					m_socketPointer->send(out);
				}
				break;
			}

			case PingNode::type:
			{
				PingNode in = PingNode::fromBytesConstRef(_from, rlpBytes);
				if (in.version < dev::p2p::c_protocolVersion)
				{
					if (in.version == 3)
					{
						compat::Pong p(in.source);
						p.echo = sha3(rlpBytes);
						p.sign(m_secret);
						m_socketPointer->send(p);
					}
					else
						return;
				}
				
				if (RLPXDatagramFace::secondsSinceEpoch() > in.ts)
				{
					clog(NodeTableTriviaSummary) << "Received expired PingNode from " << _from.address().to_string() << ":" << _from.port();
					return;
				}
				
				in.source.address = _from.address();
				in.source.udpPort = _from.port();
				addNode(Node(nodeid, in.source));
				Pong p(in.source);
				p.echo = sha3(rlpBytes);
				p.sign(m_secret);
				m_socketPointer->send(p);
				break;
			}
				
			default:
				clog(NodeTableWarn) << "Invalid message, " << hex << packetType << ", received from " << _from.address().to_string() << ":" << dec << _from.port();
				return;
		}

		noteActiveNode(nodeid, _from);
	}
	catch (...)
	{
		clog(NodeTableWarn) << "Exception processing message from " << _from.address().to_string() << ":" << _from.port();
	}
}

void NodeTable::doCheckEvictions()
{
	m_timers.schedule(c_evictionCheckInterval.count(), [this](boost::system::error_code const& _ec)
	{
		if (_ec)
			return;
		
		bool evictionsRemain = false;
		list<shared_ptr<NodeEntry>> drop;
		{
			Guard le(x_evictions);
			Guard ln(x_nodes);
			for (auto& e: m_evictions)
				if (chrono::steady_clock::now() - e.first.second > c_reqTimeout)
					if (m_nodes.count(e.second))
						drop.push_back(m_nodes[e.second]);
			evictionsRemain = (m_evictions.size() - drop.size() > 0);
		}
		
		drop.unique();
		for (auto n: drop)
			dropNode(n);
		
		if (evictionsRemain)
			doCheckEvictions();
	});
}

void NodeTable::doDiscovery()
{
	m_timers.schedule(c_bucketRefresh.count(), [this](boost::system::error_code const& ec)
	{
		if (ec)
			return;
		
		clog(NodeTableEvent) << "performing random discovery";
		NodeId randNodeId;
		crypto::Nonce::get().ref().copyTo(randNodeId.ref().cropped(0, h256::size));
		crypto::Nonce::get().ref().copyTo(randNodeId.ref().cropped(h256::size, h256::size));
		doDiscover(randNodeId);
	});
}

void PingNode::streamRLP(RLPStream& _s) const
{
	_s.appendList(4);
	_s << dev::p2p::c_protocolVersion;
	source.streamRLP(_s);
	destination.streamRLP(_s);
	_s << ts;
}

void PingNode::interpretRLP(bytesConstRef _bytes)
{
	RLP r(_bytes);
	if (r.itemCountStrict() == 4 && r[0].isInt() && r[0].toInt<unsigned>(RLP::Strict) == dev::p2p::c_protocolVersion)
	{
		version = dev::p2p::c_protocolVersion;
		source.interpretRLP(r[1]);
		destination.interpretRLP(r[2]);
		ts = r[3].toInt<uint32_t>(RLP::Strict);
	}
	else
		version = r[0].toInt<unsigned>(RLP::Strict);
}

void Pong::streamRLP(RLPStream& _s) const
{
	_s.appendList(3);
	destination.streamRLP(_s);
	_s << echo << ts;
}

void Pong::interpretRLP(bytesConstRef _bytes)
{
	RLP r(_bytes);
	destination.interpretRLP(r[0]);
	echo = (h256)r[1];
	ts = r[2].toInt<uint32_t>();
}

void compat::Pong::interpretRLP(bytesConstRef _bytes)
{
	RLP r(_bytes);
	echo = (h256)r[0];
	ts = r[1].toInt<uint32_t>();
}
