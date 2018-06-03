#pragma once

#include <boost/asio/ip/address.hpp>

#include <libethcore/Farm.h>
#include <libethcore/Miner.h>
#include <libpoolprotocols/PoolURI.h>
#include <queue>

using namespace std;

namespace dev
{
	namespace eth
	{
		class PoolClient
		{
		public:
			virtual ~PoolClient() noexcept = default;


			void setConnection(URI &conn)
			{
				m_conn = conn;
				m_connection_changed = true;
			}

			virtual void connect() = 0;
			virtual void disconnect() = 0;

			virtual void submitHashrate(string const & rate) = 0;
			virtual void submitSolution(Solution solution) = 0;
			virtual bool isConnected() = 0;
			virtual bool isPendingState() = 0;
			virtual string ActiveEndPoint() = 0;

			using SolutionAccepted = std::function<void(bool const&)>;
			using SolutionRejected = std::function<void(bool const&)>;
			using Disconnected = std::function<void()>;
			using Connected = std::function<void()>;
			using WorkReceived = std::function<void(WorkPackage const&, bool checkForDuplicates)>;

			void onSolutionAccepted(SolutionAccepted const& _handler) { m_onSolutionAccepted = _handler; }
			void onSolutionRejected(SolutionRejected const& _handler) { m_onSolutionRejected = _handler; }
			void onDisconnected(Disconnected const& _handler) { m_onDisconnected = _handler; }
			void onConnected(Connected const& _handler) { m_onConnected = _handler; }
			void onWorkReceived(WorkReceived const& _handler) { m_onWorkReceived = _handler; }

		protected:
			bool m_authorized = false;
			bool m_connected = false;
			bool m_connection_changed = false;
			boost::asio::ip::basic_endpoint<boost::asio::ip::tcp> m_endpoint;

			URI m_conn;

			SolutionAccepted m_onSolutionAccepted;
			SolutionRejected m_onSolutionRejected;
			Disconnected m_onDisconnected;
			Connected m_onConnected;
			WorkReceived m_onWorkReceived;
		};
	}
}
