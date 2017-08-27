#include "PoolManager.h"

using namespace dev;
using namespace eth;

PoolManager::PoolManager(PoolClient * client, std::map<std::string, Farm::SealerDescriptor> const & _sealers, MinerType const & minerType) 
	: m_minerType(minerType), Worker("main")
{
	p_client = client;
	p_farm = new Farm();
	p_farm->setSealers(_sealers);

	p_client->onConnected([&]()
	{
		m_reconnectTry = 0;
		cnote << "Connected to " + m_connections[m_activeConnectionIdx].host();
		if (!p_farm->isMining())
		{
			cnote << "Spinning up miners...";
			if (m_minerType == MinerType::CL)
				p_farm->start("opencl", false);
			else if (m_minerType == MinerType::CUDA)
				p_farm->start("cuda", false);
			else if (m_minerType == MinerType::Mixed) {
				p_farm->start("cuda", false);
				p_farm->start("opencl", true);
			}
		}
	});
	p_client->onDisconnected([&]()
	{
		cnote << "Disconnected from " + m_connections[m_activeConnectionIdx].host();
		if (p_farm->isMining())
		{
			cnote << "Shutting down miners...";
			p_farm->stop();
		}
		tryReconnect();
	});
	p_client->onWorkReceived([&](WorkPackage const& wp)
	{
		cnote << "Received new job #" + wp.header.hex().substr(0, 8) + " from " + m_connections[m_activeConnectionIdx].host();
		p_farm->setWork(wp);
	});
	p_client->onSolutionAccepted([&](bool const& stale)
	{
		cnote << EthLime << "B-) Submitted and accepted." << EthReset << (stale ? " (stale)" : "");
		p_farm->acceptedSolution(stale);
	});
	p_client->onSolutionRejected([&](bool const& stale)
	{
		cwarn << ":-( Not accepted." << (stale ? " (stale)" : "");
		p_farm->rejectedSolution(stale);
	});

	p_farm->onSolutionFound([&](Solution sol)
	{
		cnote << "Solution found; Submitting to " + m_connections[m_activeConnectionIdx].host() << "...";
		cnote << "  Nonce:" << toHex(sol.nonce);
		//cnote << "  headerHash:" << sol.headerHash.hex();
		//cnote << "  mixHash:" << sol.mixHash.hex();
		if (EthashAux::eval(sol.seedHash, sol.headerHash, sol.nonce).value < sol.boundary) {
			p_client->submitSolution(sol);
		}
		else {
			p_farm->failedSolution();
			cwarn << ":-( Failure: GPU gave incorrect result!";
		}
		return false;
	});
}

void PoolManager::stop()
{
	m_running = false;
	stopWorking();
}

void PoolManager::workLoop()
{
	while (m_running)
	{
		this_thread::sleep_for(chrono::seconds(1));
		m_hashrateSmoothTimePassed++;
		m_hashrateReportingTimePassed++;
		m_miningProgress = p_farm->miningProgress();

		// Hashrate reporting
		if (m_hashrateReportingTimePassed > m_hashrateReportingTime) {
			
			std::string h = toHex(toCompactBigEndian(m_miningProgress.rate(), 1));
			std::string res = h[0] != '0' ? h : h.substr(1);

			p_client->submitHashrate("0x" + res);
			m_hashrateReportingTimePassed = 0;
		}

		// Reset MiningProgress every now and then
		if (m_hashrateSmoothTimePassed > m_hashrateSmoothTime) {
			p_farm->resetMiningProgress();
			m_hashrateSmoothTimePassed = 0;
		}
	}
}

void PoolManager::addConnection(string const & host, string const & port, string const & user, string const & pass)
{
	if (host.empty()) {
		return;
	}
	PoolConnection connection(host, port, user, pass);
	m_connections.push_back(connection);

	if (m_connections.size() == 1) {
		p_client->setConnection(host, port, user, pass);
	}
}

void PoolManager::clearConnections()
{
	m_connections.clear();
	if (p_client && p_client->isConnected()) {
		p_client->disconnect();
	}
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
		cwarn << "Retrying in " << i << "... \r";
	}

	// We do not need awesome logic here, we jst have one connection anyways
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
		PoolConnection newConnection = m_connections[m_activeConnectionIdx];
		p_client->setConnection(newConnection.host(), newConnection.port(), newConnection.user(), newConnection.pass());
		p_client->connect();
	}
}
