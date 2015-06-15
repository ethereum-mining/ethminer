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
/** @file Capability.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Capability.h"

#include <libdevcore/Log.h>
#include "Session.h"
#include "Host.h"
using namespace std;
using namespace dev;
using namespace dev::p2p;

Capability::Capability(Session* _s, HostCapabilityFace* _h, unsigned _idOffset): m_session(_s), m_hostCap(_h), m_idOffset(_idOffset)
{
	clog(NetConnect) << "New session for capability" << m_hostCap->name() << "; idOffset:" << m_idOffset;
}

void Capability::disable(std::string const& _problem)
{
	clog(NetWarn) << "DISABLE: Disabling capability '" << m_hostCap->name() << "'. Reason:" << _problem;
	m_enabled = false;
}

RLPStream& Capability::prep(RLPStream& _s, unsigned _id, unsigned _args)
{
	return _s.appendRaw(bytes(1, _id + m_idOffset)).appendList(_args);
}

void Capability::sealAndSend(RLPStream& _s)
{
	m_session->sealAndSend(_s);
}

void Capability::addRating(int _r)
{
	m_session->addRating(_r);
}

ReputationManager& Capability::repMan() const
{
	return host()->repMan();
}
