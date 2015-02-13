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

NodeEntry::NodeEntry(Node _src, Public _pubk, NodeIPEndpoint _gw): Node(_pubk, _gw), distance(NodeTable::distance(_src.id,_pubk)) {}
NodeEntry::NodeEntry(Node _src, Public _pubk, bi::udp::endpoint _udp): Node(_pubk, NodeIPEndpoint(_udp)), distance(NodeTable::distance(_src.id,_pubk)) {}

NodeTable::NodeTable(ba::io_service& _io, KeyPair _alias, uint16_t _udp):
	m_node(Node(_alias.pub(), bi::udp::endpoint())),
	m_secret(_alias.sec()),
	m_io(_io),
	m_socket(new NodeSocket(m_io, *this, _udp)),
	m_socketPointer(m_socket.get()),
	m_bucketRefreshTimer(m_io),
	m_evictionCheckTimer(m_io)
{
	for (unsigned i = 0; i < s_bins; i++)
	{
		m_state[i].distance = i;
		m_state[i].modified = chrono::steady_clock::now() - chrono::seconds(1);
	}
	
	m_socketPointer->connect();
	doRefreshBuckets(boost::system::error_code());
}
	
NodeTable::~NodeTable()
{
	// Cancel scheduled tasks to ensure.
	m_evictionCheckTimer.cancel();
	m_bucketRefreshTimer.cancel();
	
	// Disconnect socket so that deallocation is safe.
	m_socketPointer->disconnect();
}

void NodeTable::processEvents()
{
	if (m_nodeEventHandler)
		m_nodeEventHandler->processEvents();
}

shared_ptr<NodeEntry> NodeTable::addNode(Public const& _pubk, bi::udp::endpoint const& _udp, bi::tcp::endpoint const& _tcp)
{
	auto node = Node(_pubk, NodeIPEndpoint(_udp, _tcp));
	return addNode(node);
}

shared_ptr<NodeEntry> NodeTable::addNode(Node const& _node)
{
	// ping address if nodeid is empty
	if (!_node.id)
	{
		PingNode p(_node.endpoint.udp, m_node.endpoint.udp.address().to_string(), m_node.endpoint.udp.port());
		p.sign(m_secret);
		m_socketPointer->send(p);
		shared_ptr<NodeEntry> n;
		return move(n);
	}
	
	Guard l(x_nodes);
	if (m_nodes.count(_node.id))
	{
//		// SECURITY: remove this in beta - it's only for lazy connections and presents an easy attack vector.
//		if (m_server->m_peers.count(id) && isPrivateAddress(m_server->m_peers.at(id)->address.address()) && ep.port() != 0)
//			// Update address if the node if we now have a public IP for it.
//			m_server->m_peers[id]->address = ep;
		return m_nodes[_node.id];
	}
	
	shared_ptr<NodeEntry> ret(new NodeEntry(m_node, _node.id, NodeIPEndpoint(_node.endpoint.udp, _node.endpoint.tcp)));
	m_nodes[_node.id] = ret;
	PingNode p(_node.endpoint.udp, m_node.endpoint.udp.address().to_string(), m_node.endpoint.udp.port());
	p.sign(m_secret);
	m_socketPointer->send(p);
	
	// TODO p2p: rename to p2p.nodes.pending, add p2p.nodes.add event (when pong is received)
	clog(NodeTableNote) << "p2p.nodes.add " << _node.id.abridged();
	if (m_nodeEventHandler)
		m_nodeEventHandler->appendEvent(_node.id, NodeEntryAdded);
	
	return ret;
}

void NodeTable::discover()
{
	static chrono::steady_clock::time_point s_lastDiscover = chrono::steady_clock::now() - std::chrono::seconds(30);
	if (chrono::steady_clock::now() > s_lastDiscover + std::chrono::seconds(30))
	{
		s_lastDiscover = chrono::steady_clock::now();
		discover(m_node.id);
	}
}

list<NodeId> NodeTable::nodes() const
{
	list<NodeId> nodes;
	Guard l(x_nodes);
	for (auto& i: m_nodes)
		nodes.push_back(i.second->id);
	return move(nodes);
}

list<NodeEntry> NodeTable::snapshot() const
{
	list<NodeEntry> ret;
	Guard l(x_state);
	for (auto s: m_state)
		for (auto n: s.nodes)
			ret.push_back(*n.lock());
	return move(ret);
}

Node NodeTable::node(NodeId const& _id)
{
	// TODO p2p: eloquent copy operator
	Guard l(x_nodes);
	if (m_nodes.count(_id))
	{
		auto entry = m_nodes[_id];
		Node n(_id, NodeIPEndpoint(entry->endpoint.udp, entry->endpoint.tcp), entry->required);
		return move(n);
	}
	return move(Node());
}

shared_ptr<NodeEntry> NodeTable::nodeEntry(NodeId _id)
{
	Guard l(x_nodes);
	return m_nodes.count(_id) ? move(m_nodes[_id]) : move(shared_ptr<NodeEntry>());
}

void NodeTable::discover(NodeId _node, unsigned _round, shared_ptr<set<shared_ptr<NodeEntry>>> _tried)
{
	if (!m_socketPointer->isOpen() || _round == s_maxSteps)
		return;
	
	if (_round == s_maxSteps)
	{
		clog(NodeTableNote) << "Terminating discover after " << _round << " rounds.";
		return;
	}
	else if(!_round && !_tried)
		// initialized _tried on first round
		_tried.reset(new set<shared_ptr<NodeEntry>>());
	
	auto nearest = nearestNodeEntries(_node);
	list<shared_ptr<NodeEntry>> tried;
	for (unsigned i = 0; i < nearest.size() && tried.size() < s_alpha; i++)
		if (!_tried->count(nearest[i]))
		{
			auto r = nearest[i];
			tried.push_back(r);
			FindNode p(r->endpoint.udp, _node);
			p.sign(m_secret);
			m_socketPointer->send(p);
		}
	
	if (tried.empty())
	{
		clog(NodeTableNote) << "Terminating discover after " << _round << " rounds.";
		return;
	}
		
	while (!tried.empty())
	{
		_tried->insert(tried.front());
		tried.pop_front();
	}
	
	auto self(shared_from_this());
	m_evictionCheckTimer.expires_from_now(boost::posix_time::milliseconds(c_reqTimeout.count()));
	m_evictionCheckTimer.async_wait([this, self, _node, _round, _tried](boost::system::error_code const& _ec)
	{
		if (_ec)
			return;
		discover(_node, _round + 1, _tried);
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
			for (auto n: m_state[head].nodes)
				if (auto p = n.lock())
				{
					if (count < s_bucketSize)
						found[distance(_target, p->id)].push_back(p);
					else
						break;
				}
			
			if (count < s_bucketSize && tail)
				for (auto n: m_state[tail].nodes)
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
			for (auto n: m_state[head].nodes)
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
			for (auto n: m_state[tail].nodes)
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
		for (auto n: nodes.second)
			ret.push_back(n);
	return move(ret);
}

void NodeTable::ping(bi::udp::endpoint _to) const
{
	PingNode p(_to, m_node.endpoint.udp.address().to_string(), m_node.endpoint.udp.port());
	p.sign(m_secret);
	m_socketPointer->send(p);
}

void NodeTable::ping(NodeEntry* _n) const
{
	if (_n)
		ping(_n->endpoint.udp);
}

void NodeTable::evict(shared_ptr<NodeEntry> _leastSeen, shared_ptr<NodeEntry> _new)
{
	if (!m_socketPointer->isOpen())
		return;
	
	{
		Guard l(x_evictions);
		m_evictions.push_back(EvictionTimeout(make_pair(_leastSeen->id,chrono::steady_clock::now()), _new->id));
		if (m_evictions.size() == 1)
			doCheckEvictions(boost::system::error_code());
		
		m_evictions.push_back(EvictionTimeout(make_pair(_leastSeen->id,chrono::steady_clock::now()), _new->id));
	}
	ping(_leastSeen.get());
}

void NodeTable::noteActiveNode(Public const& _pubk, bi::udp::endpoint const& _endpoint)
{
	if (_pubk == m_node.address())
		return;
	
	clog(NodeTableNote) << "Noting active node:" << _pubk.abridged() << _endpoint.address().to_string() << ":" << _endpoint.port();

	shared_ptr<NodeEntry> node(addNode(_pubk, _endpoint, bi::tcp::endpoint(_endpoint.address(), _endpoint.port())));

	// TODO p2p: old bug (maybe gone now) sometimes node is nullptr here
	if (!!node)
	{
		shared_ptr<NodeEntry> contested;
		{
			Guard l(x_state);
			NodeBucket& s = bucket_UNSAFE(node.get());
			s.nodes.remove_if([&node](weak_ptr<NodeEntry> n)
			{
				if (n.lock() == node)
					return true;
				return false;
			});
			
			if (s.nodes.size() >= s_bucketSize)
			{
				// It's only contested iff nodeentry exists
				contested = s.nodes.front().lock();
				if (!contested)
				{
					s.nodes.pop_front();
					s.nodes.push_back(node);
					s.touch();
				}
			}
			else
			{
				s.nodes.push_back(node);
				s.touch();
			}
		}
		
		if (contested)
			evict(contested, node);
	}
}

void NodeTable::dropNode(shared_ptr<NodeEntry> _n)
{
	{
		Guard l(x_state);
		NodeBucket& s = bucket_UNSAFE(_n.get());
		s.nodes.remove_if([&_n](weak_ptr<NodeEntry> n) { return n.lock() == _n; });
	}
	{
		Guard l(x_nodes);
		m_nodes.erase(_n->id);
	}
	
	clog(NodeTableNote) << "p2p.nodes.drop " << _n->id.abridged();
	if (m_nodeEventHandler)
		m_nodeEventHandler->appendEvent(_n->id, NodeEntryRemoved);
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
		clog(NodeTableMessageSummary) << "Invalid Message size from " << _from.address().to_string() << ":" << _from.port();
		return;
	}
	
	bytesConstRef hashedBytes(_packet.cropped(h256::size, _packet.size() - h256::size));
	h256 hashSigned(sha3(hashedBytes));
	if (!_packet.cropped(0, h256::size).contentsEqual(hashSigned.asBytes()))
	{
		clog(NodeTableMessageSummary) << "Invalid Message hash from " << _from.address().to_string() << ":" << _from.port();
		return;
	}
	
	bytesConstRef signedBytes(hashedBytes.cropped(Signature::size, hashedBytes.size() - Signature::size));

	// todo: verify sig via known-nodeid and MDC, or, do ping/pong auth if node/endpoint is unknown/untrusted
	
	bytesConstRef sigBytes(_packet.cropped(h256::size, Signature::size));
	Public nodeid(dev::recover(*(Signature const*)sigBytes.data(), sha3(signedBytes)));
	if (!nodeid)
	{
		clog(NodeTableMessageSummary) << "Invalid Message signature from " << _from.address().to_string() << ":" << _from.port();
		return;
	}
	
	unsigned packetType = signedBytes[0];
	if (packetType && packetType < 4)
		noteActiveNode(nodeid, _from);
	
	bytesConstRef rlpBytes(_packet.cropped(h256::size + Signature::size + 1));
	RLP rlp(rlpBytes);
	try {
		switch (packetType)
		{
			case Pong::type:
			{
//				clog(NodeTableMessageSummary) << "Received Pong from " << _from.address().to_string() << ":" << _from.port();
				Pong in = Pong::fromBytesConstRef(_from, rlpBytes);
				
				// whenever a pong is received, check if it's in m_evictions
				Guard le(x_evictions);
				for (auto it = m_evictions.begin(); it != m_evictions.end(); it++)
					if (it->first.first == nodeid && it->first.second > std::chrono::steady_clock::now())
					{
						if (auto n = nodeEntry(it->second))
							dropNode(n);
						
						if (auto n = node(it->first.first))
							addNode(n);
						
						it = m_evictions.erase(it);
					}
				break;
			}
				
			case Neighbours::type:
			{
				Neighbours in = Neighbours::fromBytesConstRef(_from, rlpBytes);
//				clog(NodeTableMessageSummary) << "Received " << in.nodes.size() << " Neighbours from " << _from.address().to_string() << ":" << _from.port();
				for (auto n: in.nodes)
					noteActiveNode(n.node, bi::udp::endpoint(bi::address::from_string(n.ipAddress), n.port));
				break;
			}

			case FindNode::type:
			{
//				clog(NodeTableMessageSummary) << "Received FindNode from " << _from.address().to_string() << ":" << _from.port();
				FindNode in = FindNode::fromBytesConstRef(_from, rlpBytes);

				vector<shared_ptr<NodeEntry>> nearest = nearestNodeEntries(in.target);
				static unsigned const nlimit = (m_socketPointer->maxDatagramSize - 11) / 86;
				for (unsigned offset = 0; offset < nearest.size(); offset += nlimit)
				{
					Neighbours out(_from, nearest, offset, nlimit);
					out.sign(m_secret);
					m_socketPointer->send(out);
				}
				break;
			}

			case PingNode::type:
			{
//				clog(NodeTableMessageSummary) << "Received PingNode from " << _from.address().to_string() << ":" << _from.port();
				PingNode in = PingNode::fromBytesConstRef(_from, rlpBytes);
				
				Pong p(_from);
				p.echo = sha3(rlpBytes);
				p.sign(m_secret);
				m_socketPointer->send(p);
				break;
			}
				
			default:
				clog(NodeTableWarn) << "Invalid Message, " << hex << packetType << ", received from " << _from.address().to_string() << ":" << dec << _from.port();
				return;
		}
	}
	catch (...)
	{
		clog(NodeTableWarn) << "Exception processing message from " << _from.address().to_string() << ":" << _from.port();
	}
}

void NodeTable::doCheckEvictions(boost::system::error_code const& _ec)
{
	if (_ec || !m_socketPointer->isOpen())
		return;

	auto self(shared_from_this());
	m_evictionCheckTimer.expires_from_now(c_evictionCheckInterval);
	m_evictionCheckTimer.async_wait([this, self](boost::system::error_code const& _ec)
	{
		if (_ec)
			return;
		
		bool evictionsRemain = false;
		list<shared_ptr<NodeEntry>> drop;
		{
			Guard ln(x_nodes);
			Guard le(x_evictions);
			for (auto& e: m_evictions)
				if (chrono::steady_clock::now() - e.first.second > c_reqTimeout)
					if (m_nodes.count(e.second))
						drop.push_back(m_nodes[e.second]);
			evictionsRemain = m_evictions.size() - drop.size() > 0;
		}
		
		drop.unique();
		for (auto n: drop)
			dropNode(n);
		
		if (evictionsRemain)
			doCheckEvictions(boost::system::error_code());
	});
}

void NodeTable::doRefreshBuckets(boost::system::error_code const& _ec)
{
	if (_ec)
		return;

	clog(NodeTableNote) << "refreshing buckets";
	bool connected = m_socketPointer->isOpen();
	bool refreshed = false;
	if (connected)
	{
		Guard l(x_state);
		for (auto& d: m_state)
			if (chrono::steady_clock::now() - d.modified > c_bucketRefresh)
			{
				d.touch();
				while (!d.nodes.empty())
				{
					auto n = d.nodes.front();
					if (auto p = n.lock())
					{
						refreshed = true;
						ping(p.get());
						break;
					}
					d.nodes.pop_front();
				}
			}
	}

	unsigned nextRefresh = connected ? (refreshed ? 200 : c_bucketRefresh.count()*1000) : 10000;
	auto runcb = [this](boost::system::error_code const& error) { doRefreshBuckets(error); };
	m_bucketRefreshTimer.expires_from_now(boost::posix_time::milliseconds(nextRefresh));
	m_bucketRefreshTimer.async_wait(runcb);
}

