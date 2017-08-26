#include "Farm.h"
#include "Miner.h"

using namespace std;

namespace dev
{
	namespace eth
	{
		class PoolClient
		{
		public:
			void setConnection(string const & host, string const & port = "", string const & user = "", string const & pass = "");

			virtual void connect() = 0;
			virtual void disconnect() = 0;

			virtual void submitHashrate(string const & rate) = 0;
			virtual void submitSolution(Solution solution, bool const & stale) = 0;

			bool isRunning() { return m_running; }
			bool isConnected() { return m_connected && m_authorized; }

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
			bool m_running = true;
			string m_host;
			string m_port;
			string m_user;
			string m_pass;
			bool m_connection_changed = false;

			SolutionAccepted m_onSolutionAccepted;
			SolutionRejected m_onSolutionRejected;
			Disconnected m_onDisconnected;
			Connected m_onConnected;
			WorkReceived m_onWorkReceived;
		};
	}
}