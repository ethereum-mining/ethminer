#ifndef POOL_MANAGER_H_
#define POOL_MANAGER_H_

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
		class PoolConnection
		{
		public:
			PoolConnection(string const & host, string const & port, string const & user, string const & pass) : m_host(host), m_port(port), m_user(user), m_pass(pass) {};
			string host() { return m_host; };
			string port() { return m_port; };
			string user() { return m_user; };
			string pass() { return m_pass; };
		private:
			string m_host = "";
			string m_port = "";
			string m_user = "";
			string m_pass = "";
		};

		class PoolManager : public Worker
		{
		public:
			PoolManager(PoolClient * client, Farm &farm, MinerType const & minerType);
			void addConnection(string const & host, string const & port, string const & user, string const & pass);
			void clearConnections();
			void start();
			void stop();
			void setReconnectTries(unsigned const & reconnectTries) { m_reconnectTries = reconnectTries; };
			bool isConnected() { return p_client->isConnected(); };

		private:
			unsigned m_hashrateReportingTime = 10;
			unsigned m_hashrateReportingTimePassed = 0;

			bool m_running = false;
			void workLoop() override;
			unsigned m_reconnectTries = 3;
			unsigned m_reconnectTry = 0;
			std::vector <PoolConnection> m_connections;
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

#endif

