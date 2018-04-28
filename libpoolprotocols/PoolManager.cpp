#include "PoolManager.h"
#include <chrono>

using namespace std;
using namespace dev;
using namespace eth;

static string diffToDisplay(double diff)
{
	static const char* k[] = {"hashes", "kilohashes", "megahashes", "gigahashes", "terahashes", "petahashes"};
	uint32_t i = 0;
	while ((diff > 1000.0) && (i < ((sizeof(k) / sizeof(char *)) - 2)))
	{
		i++;
		diff = diff / 1000.0;
	}
	stringstream ss;
	ss << fixed << setprecision(4) << diff << ' ' << k[i]; 
	return ss.str();
}

PoolManager::PoolManager(PoolClient * client, Farm &farm, MinerType const & minerType) : Worker("main"), m_farm(farm), m_minerType(minerType)
{
	p_client = client;

	p_client->onConnected([&]()
	{
		cnote << "Connected to " << m_connections[m_activeConnectionIdx].Host() << p_client->ActiveEndPoint();
		if (!m_farm.isMining())
		{
			cnote << "Spinning up miners...";
			if (m_minerType == MinerType::CL)
				m_farm.start("opencl", false);
			else if (m_minerType == MinerType::CUDA)
				m_farm.start("cuda", false);
			else if (m_minerType == MinerType::Mixed) {
				m_farm.start("cuda", false);
				m_farm.start("opencl", true);
			}
		}
	});
	p_client->onDisconnected([&]()
	{
		cnote << "Disconnected from " + m_connections[m_activeConnectionIdx].Host() << p_client->ActiveEndPoint();

		if (m_farm.isMining()) {
			cnote << "Shutting down miners...";
			m_farm.stop();
		}

		if (m_running)
			tryReconnect();
	});
	p_client->onWorkReceived([&](WorkPackage const& wp)
	{
		m_reconnectTry = 0;
		m_farm.setWork(wp);
		if (wp.boundary != m_lastBoundary)
		{
			using namespace boost::multiprecision;

			m_lastBoundary = wp.boundary;
			static const uint512_t dividend("0x10000000000000000000000000000000000000000000000000000000000000000");
			const uint256_t divisor(string("0x") + m_lastBoundary.hex());
			cnote << "New pool difficulty:" << EthWhite << diffToDisplay(double(dividend / divisor)) << EthReset;
		}
		cnote << "New job" << wp.header << "  " + m_connections[m_activeConnectionIdx].Host() + p_client->ActiveEndPoint();
	});
	p_client->onSolutionAccepted([&](bool const& stale)
	{
		using namespace std::chrono;
		auto ms = duration_cast<milliseconds>(steady_clock::now() - m_submit_time);
		std::stringstream ss;
		ss << std::setw(4) << std::setfill(' ') << ms.count();
		ss << "ms." << "   " << m_connections[m_activeConnectionIdx].Host() + p_client->ActiveEndPoint();
		cnote << EthLime "**Accepted" EthReset << (stale ? "(stale)" : "") << ss.str();
		m_farm.acceptedSolution(stale);
	});
	p_client->onSolutionRejected([&](bool const& stale)
	{
		using namespace std::chrono;
		auto ms = duration_cast<milliseconds>(steady_clock::now() - m_submit_time);
		std::stringstream ss;
		ss << std::setw(4) << std::setfill(' ') << ms.count();
		ss << "ms." << "   " << m_connections[m_activeConnectionIdx].Host() + p_client->ActiveEndPoint();
		cwarn << EthLime "**Rejected" EthReset << (stale ? "(stale)" : "") << ss.str();
		m_farm.rejectedSolution(stale);
	});

	m_farm.onSolutionFound([&](Solution sol)
	{
		m_submit_time = std::chrono::steady_clock::now();

		if (sol.stale)
			cnote << string(EthYellow "Stale nonce 0x") + toHex(sol.nonce);
		else
			cnote << string("Nonce 0x") + toHex(sol.nonce);

		p_client->submitSolution(sol);
		return false;
	});
	m_farm.onMinerRestart([&]() {
		dev::setThreadName("main");
		cnote << "Restart miners...";

		if (m_farm.isMining()) {
			cnote << "Shutting down miners...";
			m_farm.stop();
		}

		cnote << "Spinning up miners...";
		if (m_minerType == MinerType::CL)
			m_farm.start("opencl", false);
		else if (m_minerType == MinerType::CUDA)
			m_farm.start("cuda", false);
		else if (m_minerType == MinerType::Mixed) {
			m_farm.start("cuda", false);
			m_farm.start("opencl", true);
		}
	});
}

void PoolManager::stop()
{
	if (m_running) {
		cnote << "Shutting down...";
		m_running = false;

		if (p_client->isConnected())
			p_client->disconnect();

		if (m_farm.isMining())
		{
			cnote << "Shutting down miners...";
			m_farm.stop();
		}
	}
}

void PoolManager::workLoop()
{
	while (m_running)
	{
		this_thread::sleep_for(chrono::seconds(1));
		m_hashrateReportingTimePassed++;
		// Hashrate reporting
		if (m_hashrateReportingTimePassed > m_hashrateReportingTime) {
			auto mp = m_farm.miningProgress();
			std::string h = toHex(toCompactBigEndian(mp.rate(), 1));
			std::string res = h[0] != '0' ? h : h.substr(1);

			p_client->submitHashrate("0x" + res);
			m_hashrateReportingTimePassed = 0;
		}
	}
}

void PoolManager::addConnection(PoolConnection &conn)
{
	if (conn.Host().empty())
		return;

	m_connections.push_back(conn);

	if (m_connections.size() == 1) {
		p_client->setConnection(conn);
		m_farm.set_pool_addresses(conn.Host(), conn.Port());
	}
}

void PoolManager::clearConnections()
{
	m_connections.clear();
	m_farm.set_pool_addresses("", 0);
	if (p_client && p_client->isConnected())
		p_client->disconnect();
}

void PoolManager::start()
{
	if (m_connections.size() > 0) {
		m_running = true;
		startWorking();

		// Try to connect to pool
		p_client->connect();
	}
	else {
		cwarn << "Manager has no connections defined!";
	}
}

void PoolManager::tryReconnect()
{
	// No connections available, so why bother trying to reconnect
	if (m_connections.size() <= 0) {
		cwarn << "Manager has no connections defined!";
		return;
	}

	for (auto i = 4; --i; this_thread::sleep_for(chrono::seconds(1))) {
		cnote << "Retrying in " << i << "... \r";
	}

	// We do not need awesome logic here, we just have one connection anyway
	if (m_connections.size() == 1) {
		p_client->connect();
		return;
	}
	
	// Fallback logic, tries current connection multiple times and then switches to
	// one of the other connections.
	if (m_reconnectTries > m_reconnectTry) {
		m_reconnectTry++;
		p_client->connect();
	}
	else {
		m_reconnectTry = 0;
		m_activeConnectionIdx++;
		if (m_activeConnectionIdx >= m_connections.size()) {
			m_activeConnectionIdx = 0;
		}
		if (m_connections[m_activeConnectionIdx].Host() == "exit") {
			dev::setThreadName("main");
			cnote << "Exiting because reconnecting is not possible.";
			stop();
		}
		else {
			p_client->setConnection(m_connections[m_activeConnectionIdx]);
			m_farm.set_pool_addresses(m_connections[m_activeConnectionIdx].Host(), m_connections[m_activeConnectionIdx].Port());
			p_client->connect();
		}
	}
}
