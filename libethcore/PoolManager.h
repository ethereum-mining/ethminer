#ifndef POOL_MANAGER_H_
#define POOL_MANAGER_H_

#include "Farm.h"
#include "Miner.h"
#include "PoolClient.h"

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

		class PoolManager
		{
		public:
			PoolManager(PoolClient *client, std::map<std::string, Farm::SealerDescriptor> const& _sealers, MinerType const &minerType);
			void addConnection(string const & host, string const & port, string const & user, string const & pass);
			void clearConnections();
			void start();
			void setReconnectTries(unsigned const & reconnectTries) { m_reconnectTries = reconnectTries; };
		private:
			unsigned m_reconnectTries = 3;
			unsigned m_reconnectTry = 0;
			std::vector <PoolConnection> m_connections;
			unsigned m_activeConnectionIdx = 0;

			PoolClient *p_client;
			Farm *p_farm;
			MinerType m_minerType;

			void tryReconnect();
		};
	}
}

#endif