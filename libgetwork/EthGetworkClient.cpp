#include "EthGetworkClient.h"

using namespace std;
using namespace dev;
using namespace eth;

EthGetworkClient::EthGetworkClient(unsigned const & farmRecheckPeriod) : PoolClient()
{
	m_farmRecheckPeriod = farmRecheckPeriod;
	p_worktimer = nullptr;
	m_authorized = true;
}

EthGetworkClient::~EthGetworkClient()
{
	m_io_service.stop();
	m_serviceThread.join();
	p_client = nullptr;
	p_worktimer = nullptr;
}

void EthGetworkClient::connect()
{
	if (m_connection_changed) {
		p_client = new ::JsonrpcGetwork(new jsonrpc::HttpClient(m_host));
	}

	cnote << "connect to " << m_host;

	m_client_id = h256::random();
	m_connection_changed = false;
	m_justConnected = true; // We set a fake flag, that we can check with workhandler if connection works

	// Start getWorkThread
	if (p_worktimer) {
		p_worktimer->cancel();
	}
	p_worktimer = new boost::asio::deadline_timer(m_io_service, boost::posix_time::milliseconds(m_farmRecheckPeriod));
	p_worktimer->async_wait(boost::bind(&EthGetworkClient::getWorkHandler, this, boost::asio::placeholders::error));

	if (m_serviceThread.joinable()) {
		m_io_service.reset();
	}  else {
		m_serviceThread = std::thread{ boost::bind(&boost::asio::io_service::run, &m_io_service) };
	}
}

void EthGetworkClient::disconnect()
{
	m_connected = false;
	m_justConnected = false;

	// Stop getWorkThread
	if (p_worktimer) {
		p_worktimer->cancel();
		p_worktimer = nullptr;
	}
	m_io_service.stop();

	// Since we do not have a real connected state with getwork, we just fake it.
	if (m_onDisconnected) {
		m_onDisconnected();
	}
}

void EthGetworkClient::submitHashrate(string const & rate)
{
	if (!m_connected) {
		return;
	}

	try
	{
		p_client->eth_submitHashrate(rate, "0x" + m_client_id.hex());
	}
	catch (jsonrpc::JsonRpcException const& _e)
	{
		cwarn << "Failed to submit hashrate.";
		cwarn << boost::diagnostic_information(_e);
	}
}

void EthGetworkClient::submitSolution(Solution solution)
{
	if (!m_connected) {
		return;
	}

	try
	{
		bool accepted = p_client->eth_submitWork("0x" + toHex(solution.nonce), "0x" + toString(solution.headerHash), "0x" + toString(solution.mixHash));
		if (accepted) {
			if (m_onSolutionAccepted) {
				m_onSolutionAccepted(false);
			}
		} else {
			if (m_onSolutionRejected) {
				m_onSolutionRejected(false);
			}
		}
	}
	catch (jsonrpc::JsonRpcException const& _e)
	{
		cwarn << "Failed to submit solution.";
		cwarn << boost::diagnostic_information(_e);
	}
}

void EthGetworkClient::getWorkHandler(const boost::system::error_code& ec) {
	if (!m_connected && !m_justConnected) {
		return;
	}

	cnote << "GET WORK";

	if (!ec) {
		try
		{
			// Get Work
			Json::Value v = p_client->eth_getWork();
			WorkPackage newWorkPackage;
			newWorkPackage.header = h256(v[0].asString());
			newWorkPackage.seed = h256(v[1].asString());

			// Since we do not have a real connected state with getwork, we just fake it.
			// If getting work succeeds we know that the connection works
			if (m_justConnected && m_onConnected) {
				m_justConnected = false;
				m_onConnected();
			}
			
			// Check if header changes so the new workpackage is really new
			if (newWorkPackage.header != m_prevWorkPackage.header) {
				m_prevWorkPackage.header = newWorkPackage.header;
				m_prevWorkPackage.seed = newWorkPackage.seed;
				m_prevWorkPackage.boundary = h256(fromHex(v[2].asString()), h256::AlignRight);

				if (m_onWorkReceived) {
					m_onWorkReceived(m_prevWorkPackage);
				}
			}
			
			// Restart timer to get work again soon
			p_worktimer->expires_at(p_worktimer->expires_at() + boost::posix_time::milliseconds(m_farmRecheckPeriod));
			p_worktimer->async_wait(boost::bind(&EthGetworkClient::getWorkHandler, this, boost::asio::placeholders::error));
		}
		catch (jsonrpc::JsonRpcException)
		{
			cwarn << "Failed getting work!";
			disconnect();
		}
	}
}

