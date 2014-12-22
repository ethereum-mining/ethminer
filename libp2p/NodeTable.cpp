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

NodeTable::NodeTable(ba::io_service& _io):
		m_node(NodeEntry(Address(), Public(), bi::udp::endpoint())),
		m_socket(new nodeSocket(_io, *this, 30300)),
		m_socketPtr(m_socket.get()),
		m_io(_io),
		m_bucketRefreshTimer(m_io),
		m_evictionCheckTimer(m_io)
	{
		for (unsigned i = 0; i < s_bins; i++)
			m_state[i].distance = i, m_state[i].modified = chrono::steady_clock::now() - chrono::seconds(1);
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

NodeTable::NodeEntry NodeTable::operator[](Address _id)
{
	Guard l(x_nodes);
	return *m_nodes[_id];
}

void NodeTable::requestNeighbors(NodeEntry const& _node, Address _target) const
{
	FindNeighbors p;
	p.target = _target;
	
	p.to = _node.endpoint.udp;
	p.seal(m_secret);
	m_socketPtr->send(p);
}

void NodeTable::doFindNode(Address _node, unsigned _round, std::shared_ptr<std::set<std::shared_ptr<NodeEntry>>> _tried)
{
	if (!m_socketPtr->isOpen() || _round == 7)
		return;

	auto nearest = findNearest(_node);
	std::list<std::shared_ptr<NodeEntry>> tried;
	for (unsigned i = 0; i < nearest.size() && tried.size() < s_alpha; i++)
		if (!_tried->count(nearest[i]))
		{
			tried.push_back(nearest[i]);
			requestNeighbors(*nearest[i], _node);
		}
		else
			continue;
	
	while (auto n = tried.front())
	{
		_tried->insert(n);
		tried.pop_front();
	}
	
	auto self(shared_from_this());
	m_evictionCheckTimer.expires_from_now(boost::posix_time::milliseconds(s_findTimout));
	m_evictionCheckTimer.async_wait([this, self, _node, _round, _tried](boost::system::error_code const& _ec)
	{
		if (_ec)
			return;
		doFindNode(_node, _round + 1, _tried);
	});
}

std::vector<std::shared_ptr<NodeTable::NodeEntry>> NodeTable::findNearest(Address _target)
{
	// send s_alpha FindNeighbors packets to nodes we know, closest to target
	unsigned head = dist(m_node.id, _target);
	unsigned tail = (head - 1) % (s_bits - 1);
	
	// todo: optimize with tree
	std::map<unsigned, std::list<std::shared_ptr<NodeEntry>>> found;
	unsigned count = 0;
	
	// if d is 0, then we roll look forward, if last, we reverse, else, spread from d
	if (head != 0 && tail != s_bits)
		while (head != tail && count < s_bucketSize)
		{
			Guard l(x_state);
			for (auto& n: m_state[head].nodes)
				if (auto p = n.lock())
				{
					if (count < s_bucketSize)
						found[dist(_target, p->id)].push_back(p);
					else
						break;
				}
			
			if (count < s_bucketSize && head)
				for (auto& n: m_state[tail].nodes)
					if (auto p = n.lock())
					{
						if (count < s_bucketSize)
							found[dist(_target, p->id)].push_back(p);
						else
							break;
					}
			head++;
			tail = (tail - 1) % (s_bits - 1);
		}
	else if (head == 0)
		while (head < s_bucketSize && count < s_bucketSize)
		{
			Guard l(x_state);
			for (auto& n: m_state[head].nodes)
				if (auto p = n.lock())
				{
					if (count < s_bucketSize)
						found[dist(_target, p->id)].push_back(p);
					else
						break;
				}
			head--;
		}
	else if (tail == s_bits - 1)
		while (tail > 0 && count < s_bucketSize)
		{
			Guard l(x_state);
			for (auto& n: m_state[tail].nodes)
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
		for (auto& n: nodes.second)
			ret.push_back(n);
	return std::move(ret);
}

void NodeTable::ping(bi::address _address, unsigned _port) const
{
	PingNode p;
	string ip = m_node.endpoint.udp.address().to_string();
	p.ipAddress = asBytes(ip);
	p.port = m_node.endpoint.udp.port();
//		p.expiration;
	p.seal(m_secret);
	m_socketPtr->send(p);
}

void NodeTable::ping(NodeEntry* _n) const
{
	if (_n && _n->endpoint.udp.address().is_v4())
		ping(_n->endpoint.udp.address(), _n->endpoint.udp.port());
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

void NodeTable::noteNode(Public _pubk, bi::udp::endpoint _endpoint)
{
	Address id = right160(sha3(_pubk));
	std::shared_ptr<NodeEntry> node;
	{
		Guard l(x_nodes);
		auto n = m_nodes.find(id);
		if (n == m_nodes.end())
		{
			m_nodes[id] = std::shared_ptr<NodeEntry>(new NodeEntry(m_node, id, _pubk, _endpoint));
			node = m_nodes[id];
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
		NodeBucket s = bucket(_n.get());
		Guard l(x_state);
		s.nodes.remove_if([&_n](std::weak_ptr<NodeEntry> n)
		{
			auto p = n.lock();
			if (!p || p == _n)
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
	NodeBucket s = bucket(_n.get());
	{
		Guard l(x_state);
		s.nodes.remove_if([&_n](std::weak_ptr<NodeEntry> n) { return n.lock() == _n; });
	}
	Guard l(x_nodes);
	m_nodes.erase(_n->id);
}

NodeTable::NodeBucket const& NodeTable::bucket(NodeEntry* _n) const
{
	return m_state[_n->distance];
}

void NodeTable::onReceived(UDPSocketFace*, bi::udp::endpoint const& _from, bytesConstRef _packet)
{
	RLP rlp(_packet);
	
	
	// whenever a pong is received, first check if it's in m_evictions, if so, remove it
	Guard l(x_evictions);
}
	
void NodeTable::doCheckEvictions(boost::system::error_code const& _ec)
{
	if (_ec || !m_socketPtr->isOpen())
		return;

	m_evictionCheckTimer.expires_from_now(boost::posix_time::milliseconds(s_evictionCheckInterval));
	auto self(shared_from_this());
	m_evictionCheckTimer.async_wait([this, self](boost::system::error_code const& _ec)
	{
		if (_ec)
			return;
		
		bool evictionsRemain = false;
		std::list<shared_ptr<NodeEntry>> drop;
		{
			Guard l(x_evictions);
			for (auto& e: m_evictions)
				if (chrono::steady_clock::now() - e.first.second > chrono::milliseconds(s_pingTimeout))
				{
					Guard l(x_nodes);
					drop.push_back(m_nodes[e.second]);
				}
			evictionsRemain = m_evictions.size() - drop.size() > 0;
		}
		
		for (auto& n: drop)
			dropNode(n);
		
		if (evictionsRemain)
			doCheckEvictions(boost::system::error_code());
	});
}

void NodeTable::doRefreshBuckets(boost::system::error_code const& _ec)
{
	cout << "refreshing buckets" << endl;
	if (_ec)
		return;
	
	bool connected = m_socketPtr->isOpen();
	bool refreshed = false;
	if (connected)
	{
		Guard l(x_state);
		for (auto& d: m_state)
			if (chrono::steady_clock::now() - d.modified > chrono::seconds(s_bucketRefresh))
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

	unsigned nextRefresh = connected ? (refreshed ? 200 : s_bucketRefresh*1000) : 10000;
	auto runcb = [this](boost::system::error_code const& error) -> void { doRefreshBuckets(error); };
	m_bucketRefreshTimer.expires_from_now(boost::posix_time::milliseconds(nextRefresh));
	m_bucketRefreshTimer.async_wait(runcb);
}

