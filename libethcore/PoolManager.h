#ifndef POOL_MANAGER_H_
#define POOL_MANAGER_H_

#include <iostream>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <libdevcore/Worker.h>
#include "Farm.h"
#include "Miner.h"
#include "PoolClient.h"
#if ETH_DBUS
#include "DBusInt.h"
#endif

using namespace std;
using namespace boost::asio;

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
			PoolManager(PoolClient *client, std::map<std::string, Farm::SealerDescriptor> const& _sealers, MinerType const &minerType);
			void addConnection(string const & host, string const & port, string const & user, string const & pass);
			void clearConnections();
			void start();
			void stop();
			void setReconnectTries(unsigned const & reconnectTries) { m_reconnectTries = reconnectTries; };
			bool isConnected() { return p_client->isConnected(); };
			bool isRunning() { return p_client->isRunning(); };
			bool isMining() { return p_farm->isMining(); };
			WorkingProgress const& miningProgress() const { return m_miningProgress; };
			SolutionStats solutionStats() { return p_farm->getSolutionStats(); };

		private:
			WorkingProgress m_miningProgress;
			unsigned m_hashrateSmoothTime = 30;
			unsigned m_hashrateSmoothTimePassed = 0;

			unsigned m_hashrateReportingTime = 10;
			unsigned m_hashrateReportingTimePassed = 0;

			bool m_running = false;
			void workLoop() override;
			unsigned m_reconnectTries = 3;
			unsigned m_reconnectTry = 0;
			std::vector <PoolConnection> m_connections;
			unsigned m_activeConnectionIdx = 0;

			PoolClient *p_client;
			Farm *p_farm;
			MinerType m_minerType;

#if ETH_DBUS
			DBusInt dbusint;
#endif

			void tryReconnect();
		};
	}
}

#endif