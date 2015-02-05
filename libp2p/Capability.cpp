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
using namespace std;
using namespace dev;
using namespace dev::p2p;

#if defined(clogS)
#undef clogS
#endif
#define clogS(X) dev::LogOutputStream<X, true>(false) << "| " << std::setw(2) << session()->socketId() << "] "

Capability::Capability(Session* _s, HostCapabilityFace* _h, unsigned _idOffset): m_session(_s), m_host(_h), m_idOffset(_idOffset)
{
	clogS(NetConnect) << "New session for capability" << m_host->name() << "; idOffset:" << m_idOffset;
}

void Capability::disable(std::string const& _problem)
{
	clogS(NetWarn) << "DISABLE: Disabling capability '" << m_host->name() << "'. Reason:" << _problem;
	m_enabled = false;
}

RLPStream& Capability::prep(RLPStream& _s, unsigned _id, unsigned _args)
{
	return Session::prep(_s).appendList(_args + 1).append(_id + m_idOffset);
}

void Capability::sealAndSend(RLPStream& _s)
{
	m_session->sealAndSend(_s);
}

void Capability::send(bytesConstRef _msg)
{
	m_session->send(_msg);
}

void Capability::send(bytes&& _msg)
{
	m_session->send(move(_msg));
}

void Capability::addRating(unsigned _r)
{
	m_session->addRating(_r);
}
