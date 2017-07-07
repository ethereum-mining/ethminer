#pragma once

#include "Miner.h"

namespace dev
{
namespace eth
{

class MinerHook
{
public:
	MinerHook(Miner& _owner): m_owner(_owner) {}
	virtual ~MinerHook() = default;

	MinerHook(MinerHook const&) = delete;
	MinerHook& operator=(MinerHook const&) = delete;

	// reports progress, return true to abort
	bool found(uint64_t const* _nonces, uint32_t _count)
	{
		for (uint32_t i = 0; i < _count; ++i)
			if (m_owner.report(_nonces[i]))
				return (m_aborted = true);
		return m_owner.shouldStop();
	}

	bool searched(uint64_t _startNonce, uint32_t _count)
	{
		(void) _startNonce;
		UniqueGuard l(x_all);
		m_owner.accumulateHashes(_count);
		if (m_abort || m_owner.shouldStop())
			return (m_aborted = true);
		return false;
	}

	void abort()
	{
		{
			UniqueGuard l(x_all);
			if (m_aborted)
				return;

			m_abort = true;
		}
		// m_abort is true so now searched()/found() will return true to abort the search.
		// we hang around on this thread waiting for them to point out that they have aborted since
		// otherwise we may end up deleting this object prior to searched()/found() being called.
		m_aborted.wait(true);
	}

	void reset()
	{
		UniqueGuard l(x_all);
		m_aborted = m_abort = false;
	}

private:
	Mutex x_all;
	bool m_abort = false;
	Notified<bool> m_aborted = {true};
	Miner& m_owner;
};

}
}