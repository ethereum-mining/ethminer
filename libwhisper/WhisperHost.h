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
/** @file WhisperHost.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <mutex>
#include <array>
#include <set>
#include <memory>
#include <utility>
#include <libdevcore/RLP.h>
#include <libdevcore/Worker.h>
#include <libdevcore/Guards.h>
#include <libdevcrypto/SHA3.h>
#include "Common.h"
#include "WhisperPeer.h"
#include "Interface.h"

namespace dev
{
namespace shh
{

static const FullTopic EmptyFullTopic;

class WhisperHost: public HostCapability<WhisperPeer>, public Interface, public Worker
{
	friend class WhisperPeer;

public:
	WhisperHost();
	virtual ~WhisperHost();

	unsigned protocolVersion() const { return 2; }

	virtual void inject(Envelope const& _e, WhisperPeer* _from = nullptr) override;

	virtual FullTopic const& fullTopic(unsigned _id) const { try { return m_filters.at(m_watches.at(_id).id).full; } catch (...) { return EmptyFullTopic; } }
	virtual unsigned installWatch(FullTopic const& _filter) override;
	virtual unsigned installWatchOnId(h256 _filterId) override;
	virtual void uninstallWatch(unsigned _watchId) override;
	virtual h256s peekWatch(unsigned _watchId) const override { dev::Guard l(m_filterLock); try { return m_watches.at(_watchId).changes; } catch (...) { return h256s(); } }
	virtual h256s checkWatch(unsigned _watchId) override { cleanup(); dev::Guard l(m_filterLock); h256s ret; try { ret = m_watches.at(_watchId).changes; m_watches.at(_watchId).changes.clear(); } catch (...) {} return ret; }
	virtual h256s watchMessages(unsigned _watchId) override;

	virtual Envelope envelope(h256 _m) const override { try { dev::ReadGuard l(x_messages); return m_messages.at(_m); } catch (...) { return Envelope(); } }

	std::map<h256, Envelope> all() const { ReadGuard l(x_messages); return m_messages; }

	void cleanup();

protected:
	void doWork();

private:
	virtual void onStarting() { startWorking(); }
	virtual void onStopping() { stopWorking(); }

	void streamMessage(h256 _m, RLPStream& _s) const;

	void noteChanged(h256 _messageHash, h256 _filter);

	mutable dev::SharedMutex x_messages;
	std::map<h256, Envelope> m_messages;
	std::multimap<unsigned, h256> m_expiryQueue;

	mutable dev::Mutex m_filterLock;
	std::map<h256, InstalledFilter> m_filters;
	std::map<unsigned, ClientWatch> m_watches;
};

}
}
