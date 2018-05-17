#pragma once

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
		class PoolManager : public Worker
		{
		public:
			PoolManager(PoolClient * client, Farm &farm, MinerType const & minerType);
			void addConnection(URI &conn);
			void clearConnections();
			void start();
			void stop();
			void setReconnectTries(unsigned const & reconnectTries) { m_reconnectTries = reconnectTries; };
			bool isConnected() { return p_client->isConnected(); };
			bool isRunning() { return m_running; };

		private:
			unsigned m_hashrateReportingTime = 60;
			unsigned m_hashrateReportingTimePassed = 0;

			bool m_running = false;
			void workLoop() override;
			unsigned m_reconnectTries = 3;
			unsigned m_reconnectTry = 0;
			std::vector <URI> m_connections;
			unsigned m_activeConnectionIdx = 0;
			h256 m_lastBoundary = h256();

			PoolClient *p_client;
			Farm &m_farm;
			MinerType m_minerType;
			std::chrono::steady_clock::time_point m_submit_time;
			void tryReconnect();
		};
	}
}
