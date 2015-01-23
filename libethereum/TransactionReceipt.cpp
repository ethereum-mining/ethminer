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
/** @file TransactionReceipt.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "TransactionReceipt.h"

using namespace std;
using namespace dev;
using namespace dev::eth;

TransactionReceipt::TransactionReceipt(bytesConstRef _rlp)
{
	RLP r(_rlp);
	m_stateRoot = (h256)r[0];
	m_gasUsed = (u256)r[1];
	m_bloom = (LogBloom)r[2];
	for (auto const& i: r[3])
		m_log.emplace_back(i);
}

TransactionReceipt::TransactionReceipt(h256 _root, u256 _gasUsed, LogEntries const& _log):
	m_stateRoot(_root),
	m_gasUsed(_gasUsed),
	m_bloom(eth::bloom(_log)),
	m_log(_log)
{}

void TransactionReceipt::streamRLP(RLPStream& _s) const
{
	_s.appendList(4) << m_stateRoot << m_gasUsed << m_bloom;
	_s.appendList(m_log.size());
	for (LogEntry const& l: m_log)
		l.streamRLP(_s);
}

std::ostream& dev::eth::operator<<(std::ostream& _out, TransactionReceipt const& _r)
{
	_out << "Root: " << _r.stateRoot() << std::endl;
	_out << "Gas used: " << _r.gasUsed() << std::endl;
	_out << "Logs: " << _r.log().size() << " entries:" << std::endl;
	for (LogEntry const& i: _r.log())
	{
		_out << "Address " << i.address << ". Topics:" << std::endl;
		for (auto const& j: i.topics)
			_out << "  " << j << std::endl;
		_out << "  Data: " << toHex(i.data) << std::endl;
	}
	_out << "Bloom: " << _r.bloom() << std::endl;
	return _out;
}
