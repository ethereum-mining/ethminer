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
/** @file Capability.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include "Common.h"
#include "HostCapability.h"

namespace p2p
{

class Capability
{
	friend class Session;

public:
	Capability(Session* _s, HostCapabilityFace* _h): m_session(_s), m_host(_h) {}
	virtual ~Capability() {}

	/// Must return the capability name.
	static std::string name() { return ""; }

	Session* session() const { return m_session; }
	HostCapabilityFace* hostCapability() const { return m_host; }

protected:
	virtual bool interpret(RLP const&) = 0;

	void disable(std::string const& _problem);

	static RLPStream& prep(RLPStream& _s);
	void sealAndSend(RLPStream& _s);
	void sendDestroy(bytes& _msg);
	void send(bytesConstRef _msg);

	void addRating(unsigned _r);

private:
	Session* m_session;
	HostCapabilityFace* m_host;
	bool m_enabled = true;
};

}
