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
/** @file Common.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Miscellanea required for the Host/Session classes.
 */

#pragma once

#include "Common.h"

namespace dev
{

namespace p2p
{

class HostCapabilityFace
{
	friend class Host;
	template <class T> friend class HostCapability;
	friend class Capability;

public:
	HostCapabilityFace() {}
	virtual ~HostCapabilityFace() {}

	Host* host() const { return m_host; }

	std::vector<std::shared_ptr<Session> > peers() const;

protected:
	virtual std::string name() const = 0;
	virtual Capability* newPeerCapability(Session* _s) = 0;

	virtual void onStarting() {}
	virtual void onStopping() {}

	void seal(bytes& _b);

private:
	Host* m_host = nullptr;
};

template<class PeerCap>
class HostCapability: public HostCapabilityFace
{
public:
	HostCapability() {}
	virtual ~HostCapability() {}

	static std::string staticName() { return PeerCap::name(); }

protected:
	virtual std::string name() const { return PeerCap::name(); }
	virtual Capability* newPeerCapability(Session* _s) { return new PeerCap(_s, this); }
};

}

}
