#include "EthGetworkClient.h"

#include <ethash/ethash.hpp>

#include <chrono>

using namespace std;
using namespace dev;
using namespace eth;

EthGetworkClient::EthGetworkClient(unsigned farmRecheckPeriod, bool submitHashrate) : PoolClient(), Worker("getwork"), m_submit_hashrate(submitHashrate)
{
	m_farmRecheckPeriod = farmRecheckPeriod;
	m_authorized = true;
	m_connection_changed = true;
	if (m_submit_hashrate)
		m_client_id = h256::random();
	startWorking();
}

EthGetworkClient::~EthGetworkClient()
{
	p_client = nullptr;
}

void EthGetworkClient::connect()
{
	if (m_connection_changed) {
		stringstream ss;
		ss <<  "http://" + m_conn.Host() << ':' << m_conn.Port();
		if (m_conn.Path().length())
			ss << m_conn.Path();
		p_client = new ::JsonrpcGetwork(new jsonrpc::HttpClient(ss.str()));
	}

	m_connection_changed = false;
	m_justConnected = true; // We set a fake flag, that we can check with workhandler if connection works
}

void EthGetworkClient::disconnect()
{
	m_connected = false;
	m_justConnected = false;

	// Since we do not have a real connected state with getwork, we just fake it.
	if (m_onDisconnected) {
		m_onDisconnected();
	}
}

void EthGetworkClient::submitHashrate(string const & rate)
{
	// Store the rate in temp var. Will be handled in workLoop
	// Hashrate submission does not need to be as quick as possible
	m_currentHashrateToSubmit = rate;

}

void EthGetworkClient::submitSolution(Solution solution)
{
	// Immediately send found solution without wait for loop
	if (m_connected || m_justConnected) {
		try
		{
			bool accepted = p_client->eth_submitWork("0x" + toHex(solution.nonce), "0x" + toString(solution.work.header), "0x" + toString(solution.mixHash));
			if (accepted) {
				if (m_onSolutionAccepted) {
					m_onSolutionAccepted(false);
				}
			}
			else {
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
}

// Handles all getwork communication.
void EthGetworkClient::workLoop()
{
	while (true)
	{
		if (m_connected || m_justConnected) {

			// Get Work
			try
			{
				Json::Value v = p_client->eth_getWork();
				WorkPackage newWorkPackage;
				newWorkPackage.header = h256(v[0].asString());
                newWorkPackage.epoch = ethash::find_epoch_number(
                    ethash::hash256_from_bytes(h256{v[1].asString()}.data()));

                // Since we do not have a real connected state with getwork, we just fake it.
				// If getting work succeeds we know that the connection works
				if (m_justConnected && m_onConnected) {
					m_justConnected = false;
					m_connected = true;
					m_onConnected();
				}

				// Check if header changes so the new workpackage is really new
				if (newWorkPackage.header != m_prevWorkPackage.header) {
					m_prevWorkPackage.header = newWorkPackage.header;
					m_prevWorkPackage.epoch = newWorkPackage.epoch;
					m_prevWorkPackage.boundary = h256(fromHex(v[2].asString()), h256::AlignRight);

					if (m_onWorkReceived) {
						m_onWorkReceived(m_prevWorkPackage, true);
					}
				}
			}
			catch (const jsonrpc::JsonRpcException&)
			{
				cwarn << "Failed getting work!";
				disconnect();
			}

			// Submit current hashrate if needed
			if (m_submit_hashrate && !m_currentHashrateToSubmit.empty()) {
				try
				{
					p_client->eth_submitHashrate(m_currentHashrateToSubmit, "0x" + m_client_id.hex());
				}
				catch (const jsonrpc::JsonRpcException&)
				{
					//cwarn << "Failed to submit hashrate.";
					//cwarn << boost::diagnostic_information(_e);
				}
				m_currentHashrateToSubmit.clear();
			}
		}

		// Sleep
		this_thread::sleep_for(chrono::milliseconds(m_farmRecheckPeriod));
	}
}
