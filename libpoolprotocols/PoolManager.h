#pragma once

#include <boost/lockfree/queue.hpp>
#include <iostream>
#include <libdevcore/Worker.h>
#include <libethcore/Farm.h>
#include <libethcore/Miner.h>

#include "PoolClient.h"

#if ETH_DBUS
#include "DBusInt.h"
#endif

using namespace std;

namespace dev
{
	namespace eth
	{
		class PoolManager
		{
		public:
			PoolManager(PoolClient * client, Farm &farm, MinerType const & minerType, unsigned maxTries);
			void addConnection(URI &conn);
			void clearConnections();
			void start();
			void stop();
			bool isConnected() { return p_client->isConnected(); };
			bool isRunning() { return m_running; };

		private:
			unsigned m_hashrateReportingTime = 60;
			unsigned m_hashrateReportingTimePassed = 0;

			std::atomic<bool> m_running = { false };
			void workLoop();

			unsigned m_connectionAttempt = 0;
			unsigned m_maxConnectionAttempts = 0;
			unsigned m_activeConnectionIdx = 0;

			std::vector <URI> m_connections;
			std::thread m_workThread;

			h256 m_lastBoundary = h256();
			std::list<h256> m_headers;

			PoolClient *p_client;
			Farm &m_farm;
			MinerType m_minerType;
			boost::lockfree::queue<std::chrono::steady_clock::time_point> m_submit_times;

			int m_lastEpoch = 0;

		};
	}
}
