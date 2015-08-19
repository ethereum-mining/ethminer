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
/** @file Peer.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Peer.h"
using namespace std;
using namespace dev;
using namespace dev::p2p;

namespace dev
{

namespace p2p
{

bool Peer::shouldReconnect() const
{
	return id && endpoint && chrono::system_clock::now() > m_lastAttempted + chrono::seconds(fallbackSeconds());
}
	
unsigned Peer::fallbackSeconds() const
{
	if (peerType == PeerType::Required)
		return 5;
	switch (m_lastDisconnect)
	{
	case BadProtocol:
		return 30 * (m_failedAttempts + 1);
	case UselessPeer:
	case TooManyPeers:
		return 25 * (m_failedAttempts + 1);
	case ClientQuit:
		return 15 * (m_failedAttempts + 1);
	case NoDisconnect:
	default:
		if (m_failedAttempts < 5)
			return m_failedAttempts ? m_failedAttempts * 5 : 5;
		else if (m_failedAttempts < 15)
			return 25 + (m_failedAttempts - 5) * 10;
		else
			return 25 + 100 + (m_failedAttempts - 15) * 20;
	}
}
	
bool Peer::operator<(Peer const& _p) const
{
	if (isOffline() != _p.isOffline())
		return isOffline();
	else if (isOffline())
		if (m_lastAttempted == _p.m_lastAttempted)
			return m_failedAttempts < _p.m_failedAttempts;
		else
			return m_lastAttempted < _p.m_lastAttempted;
		else
			if (m_score == _p.m_score)
				if (m_rating == _p.m_rating)
					if (m_failedAttempts == _p.m_failedAttempts)
						return id < _p.id;
					else
						return m_failedAttempts < _p.m_failedAttempts;
					else
						return m_rating < _p.m_rating;
					else
						return m_score < _p.m_score;
}

}
}
