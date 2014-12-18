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
/** @file HostCapability.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "HostCapability.h"

#include "Session.h"
#include "Host.h"
using namespace std;
using namespace dev;
using namespace dev::p2p;

void HostCapabilityFace::seal(bytes& _b)
{
	m_host->seal(_b);
}

std::vector<std::shared_ptr<Session> > HostCapabilityFace::peers() const
{
	RecursiveGuard l(m_host->x_peers);
	std::vector<std::shared_ptr<Session> > ret;
	for (auto const& i: m_host->m_peers)
		if (std::shared_ptr<Session> p = i.second.lock())
			if (p->m_capabilities.count(capDesc()))
				ret.push_back(p);
	return ret;
}
