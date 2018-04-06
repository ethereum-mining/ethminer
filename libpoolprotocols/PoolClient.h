#pragma once

#include <boost/asio/ip/address.hpp>

#include <libethcore/Farm.h>
#include <libethcore/Miner.h>
#include <libpoolprotocols/PoolURI.h>

using namespace std;

namespace dev
{
	namespace eth
	{
		class PoolConnection
		{
		public:
			PoolConnection() {};
			PoolConnection(const URI &uri)
			  : m_host(uri.Host()),
				m_port(uri.Port()),
				m_user(uri.User()),
				m_pass(uri.Pswd()),
				m_secLevel(uri.ProtoSecureLevel()),
				m_version(uri.ProtoVersion()),
				m_path(uri.Path()) {};
			string Host() const { return m_host; };
			string Path() const { return m_path; };
			unsigned short Port() const { return m_port; };
			string User() const { return m_user; };
			string Pass() const { return m_pass; };
			SecureLevel SecLevel() const { return m_secLevel; };
			boost::asio::ip::address Address() const { return m_address; };
			unsigned Version() const { return m_version; };

			void Host(string host) { m_host = host; };
			void Path(string path) { m_path = path; };
			void Port(unsigned short port) { m_port = port; };
			void User(string user) { m_user = user; };
			void Pass(string pass) { m_pass = pass; };
			void SecLevel(SecureLevel secLevel) { m_secLevel = secLevel; };
			void Address(boost::asio::ip::address address) { m_address = address; };
			void Version(unsigned version) { m_version = version; };

		private:
			// Normally we'd replace the following with a single URI variable
			// But URI attributes are read only, and to support legacy parameters
			// we need to update these connection attributes individually.
		    string m_host;
			unsigned short m_port = 0;
			string m_user;
			string m_pass;
			SecureLevel m_secLevel = SecureLevel::NONE;
			unsigned m_version = 0;
		    string m_path;

			boost::asio::ip::address m_address;
		};

		class PoolClient
		{
		public:
			void setConnection(PoolConnection &conn)
			{
				m_conn = conn;
				m_connection_changed = true;
			}

			virtual void connect() = 0;
			virtual void disconnect() = 0;

			virtual void submitHashrate(string const & rate) = 0;
			virtual void submitSolution(Solution solution) = 0;
			virtual bool isConnected() = 0;

			using SolutionAccepted = std::function<void(bool const&)>;
			using SolutionRejected = std::function<void(bool const&)>;
			using Disconnected = std::function<void()>;
			using Connected = std::function<void()>;
			using WorkReceived = std::function<void(WorkPackage const&)>;

			void onSolutionAccepted(SolutionAccepted const& _handler) { m_onSolutionAccepted = _handler; }
			void onSolutionRejected(SolutionRejected const& _handler) { m_onSolutionRejected = _handler; }
			void onDisconnected(Disconnected const& _handler) { m_onDisconnected = _handler; }
			void onConnected(Connected const& _handler) { m_onConnected = _handler; }
			void onWorkReceived(WorkReceived const& _handler) { m_onWorkReceived = _handler; }

		protected:
			bool m_authorized = false;
			bool m_connected = false;
			PoolConnection m_conn;
			bool m_connection_changed = false;

			SolutionAccepted m_onSolutionAccepted;
			SolutionRejected m_onSolutionRejected;
			Disconnected m_onDisconnected;
			Connected m_onConnected;
			WorkReceived m_onWorkReceived;
		};
	}
}
