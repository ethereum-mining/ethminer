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
#include <libdevcore/SHA3.h>
#include "Common.h"
#include "WhisperPeer.h"
#include "Interface.h"
#include "BloomFilter.h"

namespace dev
{
namespace shh
{

static const Topics EmptyTopics;

class WhisperHost: public HostCapability<WhisperPeer>, public Interface, public Worker
{
	friend class WhisperPeer;

public:
	WhisperHost(bool _storeMessagesInDB = false);
	virtual ~WhisperHost();
	unsigned protocolVersion() const { return WhisperProtocolVersion; }
	void cleanup(); ///< remove old messages
	std::map<h256, Envelope> all() const { dev::ReadGuard l(x_messages); return m_messages; }
	TopicBloomFilterHash bloom() const { dev::Guard l(m_filterLock); return m_bloom; }

	virtual void inject(Envelope const& _e, WhisperPeer* _from = nullptr) override;
	virtual Topics const& fullTopics(unsigned _id) const override { try { return m_filters.at(m_watches.at(_id).id).full; } catch (...) { return EmptyTopics; } }
	virtual unsigned installWatch(Topics const& _filter) override;
	virtual void uninstallWatch(unsigned _watchId) override;
	virtual h256s peekWatch(unsigned _watchId) const override { dev::Guard l(m_filterLock); try { return m_watches.at(_watchId).changes; } catch (...) { return h256s(); } }
	virtual h256s checkWatch(unsigned _watchId) override;
	virtual h256s watchMessages(unsigned _watchId) override; ///< returns IDs of messages, which match specific watch criteria
	virtual Envelope envelope(h256 _m) const override { try { dev::ReadGuard l(x_messages); return m_messages.at(_m); } catch (...) { return Envelope(); } }

	void exportFilters(dev::RLPStream& o_dst) const;

protected:
	virtual void doWork() override;
	void noteAdvertiseTopicsOfInterest();
	bool isWatched(Envelope const& _e) const;

private:
	virtual void onStarting() override { startWorking(); }
	virtual void onStopping() override { stopWorking(); }
	void streamMessage(h256 _m, RLPStream& _s) const;
	void saveMessagesToBD();
	void loadMessagesFromBD();

	mutable dev::SharedMutex x_messages;
	std::map<h256, Envelope> m_messages;
	std::multimap<unsigned, h256> m_expiryQueue;

	mutable dev::Mutex m_filterLock;
	std::map<h256, InstalledFilter> m_filters;
	std::map<unsigned, ClientWatch> m_watches;
	TopicBloomFilter m_bloom;

	bool m_storeMessagesInDB; ///< needed for tests and other special cases
};

}
}
