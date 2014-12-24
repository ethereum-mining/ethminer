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

NodeTable::NodeTable(ba::io_service& _io, KeyPair _alias, uint16_t _listenPort):
	m_node(Node(_alias.address(), _alias.pub(), bi::udp::endpoint())),
	m_secret(_alias.sec()),
	m_socket(new NodeSocket(_io, *this, _listenPort)),
	m_socketPtr(m_socket.get()),
	m_io(_io),
	m_bucketRefreshTimer(m_io),
	m_evictionCheckTimer(m_io)
{
	for (unsigned i = 0; i < s_bins; i++)
		m_state[i].distance = i, m_state[i].modified = chrono::steady_clock::now() - chrono::seconds(1);
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
	
std::list<Address> NodeTable::nodes() const
{
	std::list<Address> nodes;
	Guard l(x_nodes);
	for (auto& i: m_nodes)
		nodes.push_back(i.second->id);
	return std::move(nodes);
}

list<NodeTable::NodeEntry> NodeTable::state() const
{
	list<NodeEntry> ret;
	Guard l(x_state);
	for (auto s: m_state)
		for (auto n: s.nodes)
			ret.push_back(*n.lock());
	return move(ret);
}

NodeTable::NodeEntry NodeTable::operator[](Address _id)
{
	Guard l(x_nodes);
	return *m_nodes[_id];
}

void NodeTable::requestNeighbors(NodeEntry const& _node, Address _target) const
{
	FindNode p(_node.endpoint.udp, _target);
	p.sign(m_secret);
	m_socketPtr->send(p);
}

void NodeTable::doFindNode(Address _node, unsigned _round, std::shared_ptr<std::set<std::shared_ptr<NodeEntry>>> _tried)
{
	if (!m_socketPtr->isOpen() || _round == s_maxSteps)
		return;
	
	if (_round == s_maxSteps)
	{
		clog(NodeTableWarn) << "Terminating doFindNode after " << _round << " rounds.";
		return;
	}
	else
		_tried.reset(new std::set<std::shared_ptr<NodeEntry>>());
	
	auto nearest = findNearest(_node);
	std::list<std::shared_ptr<NodeEntry>> tried;
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
		clog(NodeTableWarn) << "Terminating doFindNode after " << _round << " rounds.";
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

std::vector<std::shared_ptr<NodeTable::NodeEntry>> NodeTable::findNearest(Address _target)
{
	// send s_alpha FindNode packets to nodes we know, closest to target
	static unsigned lastBin = s_bins - 1;
	unsigned head = dist(m_node.id, _target);
	unsigned tail = head == 0 ? lastBin : (head - 1) % s_bins;
	
	std::map<unsigned, std::list<std::shared_ptr<NodeEntry>>> found;
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
	
	std::vector<std::shared_ptr<NodeEntry>> ret;
	for (auto& nodes: found)
		for (auto n: nodes.second)
			ret.push_back(n);
	return std::move(ret);
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

void NodeTable::evict(std::shared_ptr<NodeEntry> _leastSeen, std::shared_ptr<NodeEntry> _new)
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

void NodeTable::noteNode(Public const& _pubk, bi::udp::endpoint const& _endpoint)
{
	Address id = right160(sha3(_pubk));
	
	// Don't add ourself (would result in -1 bucket lookup)
	if (id == m_node.address())
		return;
	
	std::shared_ptr<NodeEntry> node;
	{
		Guard l(x_nodes);
		auto n = m_nodes.find(id);
		if (n == m_nodes.end())
		{
			node.reset(new NodeEntry(m_node, id, _pubk, _endpoint));
			m_nodes[id] = node;
		}
		else
			node = n->second;
	}
	
	noteNode(node);
}

void NodeTable::noteNode(std::shared_ptr<NodeEntry> _n)
{
	std::shared_ptr<NodeEntry> contested;
	{
		NodeBucket& s = bucket(_n.get());
		Guard l(x_state);
		s.nodes.remove_if([&_n](std::weak_ptr<NodeEntry> n)
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

void NodeTable::dropNode(std::shared_ptr<NodeEntry> _n)
{
	NodeBucket &s = bucket(_n.get());
	{
		Guard l(x_state);
		s.nodes.remove_if([&_n](std::weak_ptr<NodeEntry> n) { return n.lock() == _n; });
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
	if (_packet.size() < 69)
	{
		clog(NodeTableMessageSummary) << "Invalid Message received from " << _from.address().to_string() << ":" << _from.port();
		return;
	}
	
	// 3 items is PingNode, 2 items w/no lists is FindNode, 2 items w/first item as list is Neighbors, 1 item is Pong
	bytesConstRef rlpBytes(_packet.cropped(65, _packet.size() - 65));
	RLP rlp(rlpBytes);
	unsigned itemCount = rlp.itemCount();
	
	bytesConstRef sigBytes(_packet.cropped(0, 65));
	Public nodeid(dev::recover(*(Signature const*)sigBytes.data(), sha3(rlpBytes)));
	if (!nodeid)
	{
		clog(NodeTableMessageSummary) << "Invalid Message Signature from " << _from.address().to_string() << ":" << _from.port();
		return;
	}
	noteNode(nodeid, _from);
	
	try {
		switch (itemCount) {
			case 1:
			{
				clog(NodeTableMessageSummary) << "Received Pong from " << _from.address().to_string() << ":" << _from.port();
				Pong in = Pong::fromBytesConstRef(_from, rlpBytes);
				
				// whenever a pong is received, first check if it's in m_evictions
				
				break;
			}
				
			case 2:
				if (rlp[0].isList())
				{
					// todo: chunk neighbors packet
					clog(NodeTableMessageSummary) << "Received Neighbors from " << _from.address().to_string() << ":" << _from.port();
					Neighbors in = Neighbors::fromBytesConstRef(_from, rlpBytes);
					for (auto n: in.nodes)
						noteNode(n.node, bi::udp::endpoint(bi::address::from_string(n.ipAddress), n.port));
				}
				else
				{
					clog(NodeTableMessageSummary) << "Received FindNode from " << _from.address().to_string() << ":" << _from.port();
					FindNode in = FindNode::fromBytesConstRef(_from, rlpBytes);
					
					std::vector<std::shared_ptr<NodeTable::NodeEntry>> nearest = findNearest(in.target);
					Neighbors out(_from, nearest);
					out.sign(m_secret);
					m_socketPtr->send(out);
				}
				break;
				
			case 3:
			{
				clog(NodeTableMessageSummary) << "Received PingNode from " << _from.address().to_string() << ":" << _from.port();
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
		std::list<shared_ptr<NodeEntry>> drop;
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

