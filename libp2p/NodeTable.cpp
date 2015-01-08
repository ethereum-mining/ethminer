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

NodeEntry::NodeEntry(Node _src, Public _pubk, NodeIPEndpoint _gw): Node(_pubk, _gw), distance(NodeTable::dist(_src.id,_pubk)) {}
NodeEntry::NodeEntry(Node _src, Public _pubk, bi::udp::endpoint _udp): Node(_pubk, NodeIPEndpoint(_udp)), distance(NodeTable::dist(_src.id,_pubk)) {}

NodeTable::NodeTable(ba::io_service& _io, KeyPair _alias, uint16_t _udp, bi::tcp::endpoint _ep):
	m_node(Node(_alias.pub(), bi::udp::endpoint())),
	m_secret(_alias.sec()),
	m_io(_io),
	m_socket(new NodeSocket(m_io, *this, _udp)),
	m_socketPtr(m_socket.get()),
	m_bucketRefreshTimer(m_io),
	m_evictionCheckTimer(m_io)
{
	for (unsigned i = 0; i < s_bins; i++)
	{
		m_state[i].distance = i;
		m_state[i].modified = chrono::steady_clock::now() - chrono::seconds(1);
	}
	
	m_socketPtr->connect();
	doRefreshBuckets(boost::system::error_code());
}
	
NodeTable::~NodeTable()
{
	m_evictionCheckTimer.cancel();
	m_bucketRefreshTimer.cancel();
	m_socketPtr->disconnect();
}

void NodeTable::join()
{
	doFindNode(m_node.id);
}
	
list<NodeId> NodeTable::nodes() const
{
	list<NodeId> nodes;
	Guard l(x_nodes);
	for (auto& i: m_nodes)
		nodes.push_back(i.second->id);
	return move(nodes);
}

list<NodeEntry> NodeTable::state() const
{
	list<NodeEntry> ret;
	Guard l(x_state);
	for (auto s: m_state)
		for (auto n: s.nodes)
			ret.push_back(*n.lock());
	return move(ret);
}

Node NodeTable::operator[](NodeId _id)
{
	Guard l(x_nodes);
	auto n = m_nodes[_id];
	return !!n ? *n : Node();
}

void NodeTable::requestNeighbours(NodeEntry const& _node, NodeId _target) const
{
	FindNode p(_node.endpoint.udp, _target);
	p.sign(m_secret);
	m_socketPtr->send(p);
}

void NodeTable::doFindNode(NodeId _node, unsigned _round, shared_ptr<set<shared_ptr<NodeEntry>>> _tried)
{
	if (!m_socketPtr->isOpen() || _round == s_maxSteps)
		return;
	
	if (_round == s_maxSteps)
	{
		clog(NodeTableNote) << "Terminating doFindNode after " << _round << " rounds.";
		return;
	}
	else if(!_round && !_tried)
		// initialized _tried on first round
		_tried.reset(new set<shared_ptr<NodeEntry>>());
	
	auto nearest = findNearest(_node);
	list<shared_ptr<NodeEntry>> tried;
	for (unsigned i = 0; i < nearest.size() && tried.size() < s_alpha; i++)
		if (!_tried->count(nearest[i]))
		{
			auto r = nearest[i];
			tried.push_back(r);
			FindNode p(r->endpoint.udp, _node);
			p.sign(m_secret);
			m_socketPtr->send(p);
		}
	
	if (tried.empty())
	{
		clog(NodeTableNote) << "Terminating doFindNode after " << _round << " rounds.";
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
		doFindNode(_node, _round + 1, _tried);
	});
}

vector<shared_ptr<NodeEntry>> NodeTable::findNearest(NodeId _target)
{
	// send s_alpha FindNode packets to nodes we know, closest to target
	static unsigned lastBin = s_bins - 1;
	unsigned head = dist(m_node.id, _target);
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
						found[dist(_target, p->id)].push_back(p);
					else
						break;
				}
			
			if (count < s_bucketSize && tail)
				for (auto n: m_state[tail].nodes)
					if (auto p = n.lock())
					{
						if (count < s_bucketSize)
							found[dist(_target, p->id)].push_back(p);
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
						found[dist(_target, p->id)].push_back(p);
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
						found[dist(_target, p->id)].push_back(p);
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
	m_socketPtr->send(p);
}

void NodeTable::ping(NodeEntry* _n) const
{
	if (_n)
		ping(_n->endpoint.udp);
}

void NodeTable::evict(shared_ptr<NodeEntry> _leastSeen, shared_ptr<NodeEntry> _new)
{
	if (!m_socketPtr->isOpen())
		return;
	
	Guard l(x_evictions);
	m_evictions.push_back(EvictionTimeout(make_pair(_leastSeen->id,chrono::steady_clock::now()), _new->id));
	if (m_evictions.size() == 1)
		doCheckEvictions(boost::system::error_code());
	
	m_evictions.push_back(EvictionTimeout(make_pair(_leastSeen->id,chrono::steady_clock::now()), _new->id));
	ping(_leastSeen.get());
}

shared_ptr<NodeEntry> NodeTable::addNode(Public const& _pubk, bi::udp::endpoint const& _udp, bi::tcp::endpoint const& _tcp)
{
	auto node = Node(_pubk, NodeIPEndpoint(_udp, _tcp));
	return move(addNode(node));
}

shared_ptr<NodeEntry> NodeTable::addNode(Node const& _node)
{
	shared_ptr<NodeEntry> ret;
	Guard l(x_nodes);
	if (auto n = m_nodes[_node.id])
		ret = n;
	else
	{
		ret.reset(new NodeEntry(m_node, _node.id, NodeIPEndpoint(_node.endpoint.udp, _node.endpoint.tcp)));
		m_nodes[_node.id] = ret;
		PingNode p(_node.endpoint.udp, m_node.endpoint.udp.address().to_string(), m_node.endpoint.udp.port());
		p.sign(m_secret);
		m_socketPtr->send(p);
	}
	return move(ret);
}

void NodeTable::noteNode(Public const& _pubk, bi::udp::endpoint const& _endpoint)
{
	if (_pubk == m_node.address())
		return;
	
	shared_ptr<NodeEntry> node(addNode(_pubk, _endpoint));

	// todo: sometimes node is nullptr here
	if (!!node)
		noteNode(node);
}

void NodeTable::noteNode(shared_ptr<NodeEntry> _n)
{
	shared_ptr<NodeEntry> contested;
	{
		NodeBucket& s = bucket(_n.get());
		Guard l(x_state);
		s.nodes.remove_if([&_n](weak_ptr<NodeEntry> n)
		{
			if (n.lock() == _n)
				return true;
			return false;
		});

		if (s.nodes.size() >= s_bucketSize)
		{
			contested = s.nodes.front().lock();
			if (!contested)
			{
				s.nodes.pop_front();
				s.nodes.push_back(_n);
			}
		}
		else
			s.nodes.push_back(_n);
	}
	
	if (contested)
		evict(contested, _n);
}

void NodeTable::dropNode(shared_ptr<NodeEntry> _n)
{
	NodeBucket &s = bucket(_n.get());
	{
		Guard l(x_state);
		s.nodes.remove_if([&_n](weak_ptr<NodeEntry> n) { return n.lock() == _n; });
	}
	Guard l(x_nodes);
	m_nodes.erase(_n->id);
}

NodeTable::NodeBucket& NodeTable::bucket(NodeEntry const* _n)
{
	return m_state[_n->distance - 1];
}

void NodeTable::onReceived(UDPSocketFace*, bi::udp::endpoint const& _from, bytesConstRef _packet)
{
	// h256 + Signature + RLP (smallest possible packet is empty neighbours packet which is 3 bytes)
	if (_packet.size() < h256::size + Signature::size + 3)
	{
		clog(NodeTableMessageSummary) << "Invalid Message size from " << _from.address().to_string() << ":" << _from.port();
		return;
	}
	
	bytesConstRef signedBytes(_packet.cropped(h256::size, _packet.size() - h256::size));
	h256 hashSigned(sha3(signedBytes));
	if (!_packet.cropped(0, h256::size).contentsEqual(hashSigned.asBytes()))
	{
		clog(NodeTableMessageSummary) << "Invalid Message hash from " << _from.address().to_string() << ":" << _from.port();
		return;
	}

	bytesConstRef rlpBytes(signedBytes.cropped(Signature::size, signedBytes.size() - Signature::size));
	RLP rlp(rlpBytes);
	unsigned itemCount = rlp.itemCount();
	
	bytesConstRef sigBytes(_packet.cropped(h256::size, Signature::size));
	Public nodeid(dev::recover(*(Signature const*)sigBytes.data(), sha3(rlpBytes)));
	if (!nodeid)
	{
		clog(NodeTableMessageSummary) << "Invalid Message signature from " << _from.address().to_string() << ":" << _from.port();
		return;
	}
	noteNode(nodeid, _from);
	
	try {
		switch (itemCount)
		{
			case 1:
			{
//				clog(NodeTableMessageSummary) << "Received Pong from " << _from.address().to_string() << ":" << _from.port();
				Pong in = Pong::fromBytesConstRef(_from, rlpBytes);
				
				// whenever a pong is received, first check if it's in m_evictions
				
				break;
			}
				
			case 2:
				if (rlp[0].isList())
				{
					Neighbours in = Neighbours::fromBytesConstRef(_from, rlpBytes);
//					clog(NodeTableMessageSummary) << "Received " << in.nodes.size() << " Neighbours from " << _from.address().to_string() << ":" << _from.port();
					for (auto n: in.nodes)
						noteNode(n.node, bi::udp::endpoint(bi::address::from_string(n.ipAddress), n.port));
				}
				else
				{
//					clog(NodeTableMessageSummary) << "Received FindNode from " << _from.address().to_string() << ":" << _from.port();
					FindNode in = FindNode::fromBytesConstRef(_from, rlpBytes);
					
					vector<shared_ptr<NodeEntry>> nearest = findNearest(in.target);
					static unsigned const nlimit = (m_socketPtr->maxDatagramSize - 11) / 86;
					for (unsigned offset = 0; offset < nearest.size(); offset += nlimit)
					{
						Neighbours out(_from, nearest, offset, nlimit);
						out.sign(m_secret);
						m_socketPtr->send(out);
					}
				}
				break;
				
			case 3:
			{
//				clog(NodeTableMessageSummary) << "Received PingNode from " << _from.address().to_string() << ":" << _from.port();
				PingNode in = PingNode::fromBytesConstRef(_from, rlpBytes);
				
				Pong p(_from);
				p.replyTo = sha3(rlpBytes);
				p.sign(m_secret);
				m_socketPtr->send(p);
				break;
			}
				
			default:
				clog(NodeTableMessageSummary) << "Invalid Message received from " << _from.address().to_string() << ":" << _from.port();
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
	if (_ec || !m_socketPtr->isOpen())
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
			Guard le(x_evictions);
			Guard ln(x_nodes);
			for (auto& e: m_evictions)
				if (chrono::steady_clock::now() - e.first.second > c_reqTimeout)
					if (auto n = m_nodes[e.second])
						drop.push_back(n);
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
	bool connected = m_socketPtr->isOpen();
	bool refreshed = false;
	if (connected)
	{
		Guard l(x_state);
		for (auto& d: m_state)
			if (chrono::steady_clock::now() - d.modified > c_bucketRefresh)
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

	unsigned nextRefresh = connected ? (refreshed ? 200 : c_bucketRefresh.count()*1000) : 10000;
	auto runcb = [this](boost::system::error_code const& error) -> void { doRefreshBuckets(error); };
	m_bucketRefreshTimer.expires_from_now(boost::posix_time::milliseconds(nextRefresh));
	m_bucketRefreshTimer.async_wait(runcb);
}

