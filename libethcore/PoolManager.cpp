#include "PoolManager.h"

using namespace dev;
using namespace eth;

PoolManager::PoolManager(PoolClient * client, std::map<std::string, Farm::SealerDescriptor> const & _sealers, MinerType const & minerType) : m_minerType(minerType)
{
	p_client = client;
	p_farm = new Farm();
	p_farm->setSealers(_sealers);

	p_client->onConnected([&]()
	{
		// handle startup of GPU workers here (aka farm.start)
		cnote << "onConnected";
	});
	p_client->onDisconnected([&]()
	{
		cnote << "onDisconnected";
		tryReconnect();
	});
	p_client->onWorkReceived([&](WorkPackage const& wp) {
		// handle setting package to GPU workers (aka farm.setWork)
		cnote << "Got new package from POOL";
	});
	p_client->onSolutionAccepted([&](bool const& stale)
	{
		// handle increase of counters
		cnote << "SolutionAccepted" << stale;
	});
	p_client->onSolutionRejected([&](bool const& stale)
	{
		// handle increase of counters
		cnote << "SolutionRejected" << stale;
	});

	p_farm->onSolutionFound([&](Solution sol)
	{
		// actually check if solution is valid or stale
		p_client->submitSolution(sol, false);
		return false;
	});
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
}


void PoolManager::start()
{
	if (m_connections.size() > 0) {
		p_client->connect();
	}
	else {
		cwarn << "Manager has no connections defined!";
	}
}

void PoolManager::tryReconnect()
{
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
