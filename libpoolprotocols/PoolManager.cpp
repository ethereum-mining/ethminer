#include <chrono>

#include "PoolManager.h"

using namespace std;
using namespace dev;
using namespace eth;

PoolManager* PoolManager::m_this = nullptr;

PoolManager::PoolManager(PoolSettings _settings)
  : m_Settings(std::move(_settings)),
    m_io_strand(g_io_service),
    m_failovertimer(g_io_service),
    m_submithrtimer(g_io_service),
    m_reconnecttimer(g_io_service)
{
<<<<<<< HEAD
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "PoolManager::PoolManager() begin");

    m_this = this;

    m_currentWp.header = h256();

    Farm::f().onMinerRestart([&]() {
        cnote << "Restart miners...";

        if (Farm::f().isMining())
        {
            cnote << "Shutting down miners...";
            Farm::f().stop();
        }

        cnote << "Spinning up miners...";
        Farm::f().start();
    });

    Farm::f().onSolutionFound([&](const Solution& sol) {
        // Solution should passthrough only if client is
        // properly connected. Otherwise we'll have the bad behavior
        // to log nonce submission but receive no response

        if (p_client && p_client->isConnected())
        {
            p_client->submitSolution(sol);
        }
        else
        {
            cnote << string(EthOrange "Solution 0x") + toHex(sol.nonce)
                  << " wasted. Waiting for connection...";
        }

        return false;
    });


    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "PoolManager::PoolManager() end");
=======
	static const char* k[] = {"hashes", "kilohashes", "megahashes", "gigahashes", "terahashes", "petahashes"};
	uint32_t i = 0;
	while ((diff > 1000.0) && (i < ((sizeof(k) / sizeof(char *)) - 2)))
	{
		i++;
		diff = diff / 1000.0;
	}
	stringstream ss;
	ss << fixed << setprecision(2) << diff << ' ' << k[i];
	return ss.str();
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
}

void PoolManager::setClientHandlers()
{
<<<<<<< HEAD
    p_client->onConnected([&]() {
        {

            // If HostName is already an IP address no need to append the
            // effective ip address.
            if (p_client->getConnection()->HostNameType() == dev::UriHostNameType::Dns ||
                p_client->getConnection()->HostNameType() == dev::UriHostNameType::Basic)
            {
                string ep = p_client->ActiveEndPoint();
                if (!ep.empty())
                    m_selectedHost = p_client->getConnection()->Host() + ep;
            }

            cnote << "Established connection to " << m_selectedHost;
            m_connectionAttempt = 0;

            // Reset current WorkPackage
            m_currentWp.job.clear();
            m_currentWp.header = h256();

            // Shuffle if needed
            if (Farm::f().get_ergodicity() == 1U)
                Farm::f().shuffle();

            // Rough implementation to return to primary pool
            // after specified amount of time
            if (m_activeConnectionIdx != 0 && m_Settings.poolFailoverTimeout)
            {
                m_failovertimer.expires_from_now(
                    boost::posix_time::minutes(m_Settings.poolFailoverTimeout));
                m_failovertimer.async_wait(m_io_strand.wrap(boost::bind(
                    &PoolManager::failovertimer_elapsed, this, boost::asio::placeholders::error)));
            }
            else
            {
                m_failovertimer.cancel();
            }
        }

        if (!Farm::f().isMining())
        {
            cnote << "Spinning up miners...";
            Farm::f().start();
        }
        else if (Farm::f().paused())
        {
            cnote << "Resume mining ...";
            Farm::f().resume();
        }

        // Activate timing for HR submission
        if (m_Settings.reportHashrate)
        {
            m_submithrtimer.expires_from_now(boost::posix_time::seconds(m_Settings.hashRateInterval));
            m_submithrtimer.async_wait(m_io_strand.wrap(boost::bind(
                &PoolManager::submithrtimer_elapsed, this, boost::asio::placeholders::error)));
        }

        // Signal async operations have completed
        m_async_pending.store(false, std::memory_order_relaxed);

    });

    p_client->onDisconnected([&]() {
        cnote << "Disconnected from " << m_selectedHost;

        // Clear current connection
        p_client->unsetConnection();
        m_currentWp.header = h256();

        // Stop timing actors
        m_failovertimer.cancel();
        m_submithrtimer.cancel();

        if (m_stopping.load(std::memory_order_relaxed))
        {
            if (Farm::f().isMining())
            {
                cnote << "Shutting down miners...";
                Farm::f().stop();
            }
            m_running.store(false, std::memory_order_relaxed);
        }
        else
        {
            // Signal we will reconnect async
            m_async_pending.store(true, std::memory_order_relaxed);

            // Suspend mining and submit new connection request
            cnote << "No connection. Suspend mining ...";
            Farm::f().pause();
            g_io_service.post(m_io_strand.wrap(boost::bind(&PoolManager::rotateConnect, this)));
        }
    });

    p_client->onWorkReceived([&](WorkPackage const& wp) {

        // Should not happen !
        if (!wp)
            return;

        int _currentEpoch = m_currentWp.epoch;
        bool newEpoch = (_currentEpoch == -1);

        // In EthereumStratum/2.0.0 epoch number is set in session
        if (!newEpoch)
        {
            if (p_client->getConnection()->StratumMode() == 3)
                newEpoch = (wp.epoch != m_currentWp.epoch);
            else
                newEpoch = (wp.seed != m_currentWp.seed);
        }

        bool newDiff = (wp.boundary != m_currentWp.boundary);

        m_currentWp = wp;

        if (newEpoch)
        {
            m_epochChanges.fetch_add(1, std::memory_order_relaxed);

            // If epoch is valued in workpackage take it
            if (wp.epoch == -1)
            {
                if (m_currentWp.block >= 0)
                    m_currentWp.epoch = m_currentWp.block / 30000;
                else
                    m_currentWp.epoch = ethash::find_epoch_number(
                        ethash::hash256_from_bytes(m_currentWp.seed.data()));
            }
        }
        else
        {
            m_currentWp.epoch = _currentEpoch;
        }

        if (newDiff || newEpoch)
            showMiningAt();

        cnote << "Job: " EthWhite << m_currentWp.header.abridged()
              << (m_currentWp.block != -1 ? (" block " + to_string(m_currentWp.block)) : "")
              << EthReset << " " << m_selectedHost;

        Farm::f().setWork(m_currentWp);
    });

    p_client->onSolutionAccepted(
        [&](std::chrono::milliseconds const& _responseDelay, unsigned const& _minerIdx, bool _asStale) {
            std::stringstream ss;
            ss << std::setw(4) << std::setfill(' ') << _responseDelay.count() << " ms. "
               << m_selectedHost;
            cnote << EthLime "**Accepted" << (_asStale ? " stale": "") << EthReset << ss.str();
            Farm::f().accountSolution(_minerIdx, SolutionAccountingEnum::Accepted);
        });

    p_client->onSolutionRejected(
        [&](std::chrono::milliseconds const& _responseDelay, unsigned const& _minerIdx) {
            std::stringstream ss;
            ss << std::setw(4) << std::setfill(' ') << _responseDelay.count() << " ms. "
               << m_selectedHost;
            cwarn << EthRed "**Rejected" EthReset << ss.str();
            Farm::f().accountSolution(_minerIdx, SolutionAccountingEnum::Rejected);
        });
=======
	p_client = client;

	p_client->onConnected([&]()
	{
		cnote << "Connected to " << m_connections[m_activeConnectionIdx].Host() << p_client->ActiveEndPoint();
		if (!m_farm.isMining())
		{
			cnote << "Spinning up miners...";
			if (m_minerType == MinerType::CL)
				m_farm.start("opencl", false);
			else if (m_minerType == MinerType::Fpga)
				m_farm.start("fpga", false);
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
		cwarn << EthRed "**Rejected" EthReset << (stale ? "(stale)" : "") << ss.str();
		m_farm.rejectedSolution(stale);
	});

	m_farm.onSolutionFound([&](Solution sol)
	{
		// Solution should passthrough only if client is
		// properly connected. Otherwise we'll have the bad behavior
		// to log nonce submission but receive no response

		if (p_client->isConnected()) {

			m_submit_time = std::chrono::steady_clock::now();

			if (sol.stale)
				cnote << string(EthYellow "Stale nonce 0x") + toHex(sol.nonce);
			else
				cnote << string("Nonce 0x") + toHex(sol.nonce);

			p_client->submitSolution(sol);

		}
		else {

			cnote << string(EthRed "Nonce 0x") + toHex(sol.nonce) << "wasted. Waiting for connection ...";

		}

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
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
}

void PoolManager::stop()
{
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "PoolManager::stop() begin");
    if (m_running.load(std::memory_order_relaxed))
    {
        m_async_pending.store(true, std::memory_order_relaxed);
        m_stopping.store(true, std::memory_order_relaxed);

        if (p_client && p_client->isConnected())
        {
            p_client->disconnect();
            // Wait for async operations to complete
            while (m_running.load(std::memory_order_relaxed))
                this_thread::sleep_for(chrono::milliseconds(500));

            p_client = nullptr;
        }
        else
        {
            // Stop timing actors
            m_failovertimer.cancel();
            m_submithrtimer.cancel();
            m_reconnecttimer.cancel();

            if (Farm::f().isMining())
            {
                cnote << "Shutting down miners...";
                Farm::f().stop();
            }
        }
    }
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "PoolManager::stop() end");
}

void PoolManager::addConnection(std::string _connstring)
{
    m_Settings.connections.push_back(std::shared_ptr<URI>(new URI(_connstring)));
}

void PoolManager::addConnection(std::shared_ptr<URI> _uri)
{
    m_Settings.connections.push_back(_uri);
}

/*
 * Remove a connection
 * Returns:  0 on success
 *          -1 failure (out of bounds)
 *          -2 failure (active connection should be deleted)
 */
void PoolManager::removeConnection(unsigned int idx)
{
    // Are there any outstanding operations ?
    if (m_async_pending.load(std::memory_order_relaxed))
        throw std::runtime_error("Outstanding operations. Retry ...");

    // Check bounds
    if (idx >= m_Settings.connections.size())
        throw std::runtime_error("Index out-of bounds.");

    // Can't delete active connection
    if (idx == m_activeConnectionIdx)
        throw std::runtime_error("Can't remove active connection");

    // Remove the selected connection
    m_Settings.connections.erase(m_Settings.connections.begin() + idx);
    if (m_activeConnectionIdx > idx)
        m_activeConnectionIdx--;

}

void PoolManager::setActiveConnectionCommon(unsigned int idx)
{

    // Are there any outstanding operations ?
    bool ex = false;
    if (!m_async_pending.compare_exchange_strong(ex, true))
        throw std::runtime_error("Outstanding operations. Retry ...");

    if (idx != m_activeConnectionIdx)
    {
        m_connectionSwitches.fetch_add(1, std::memory_order_relaxed);
        m_activeConnectionIdx = idx;
        m_connectionAttempt = 0;
        p_client->disconnect();
    }
    else
    {
        // Release the flag immediately
        m_async_pending.store(false, std::memory_order_relaxed);
    }

}

/*
 * Sets the active connection
 * Returns: 0 on success, -1 on failure (out of bounds)
 */
void PoolManager::setActiveConnection(unsigned int idx)
{
    // Sets the active connection to the requested index
    if (idx >= m_Settings.connections.size())
        throw std::runtime_error("Index out-of bounds.");

    setActiveConnectionCommon(idx);
}

void PoolManager::setActiveConnection(std::string& _connstring)
{
    for (size_t idx = 0; idx < m_Settings.connections.size(); idx++)
        if (boost::iequals(m_Settings.connections[idx]->str(), _connstring))
        {
            setActiveConnectionCommon(idx);
            return;
        }
    throw std::runtime_error("Not found.");
}

std::shared_ptr<URI> PoolManager::getActiveConnection()
{
    try
    {
        return m_Settings.connections.at(m_activeConnectionIdx);
    }
    catch (const std::exception&)
    {
        return nullptr;
    }
}

Json::Value PoolManager::getConnectionsJson()
{
    // Returns the list of configured connections
    Json::Value jRes;
    for (size_t i = 0; i < m_Settings.connections.size(); i++)
    {
        Json::Value JConn;
        JConn["index"] = (unsigned)i;
        JConn["active"] = (i == m_activeConnectionIdx ? true : false);
        JConn["uri"] = m_Settings.connections[i]->str();
        jRes.append(JConn);
    }
    return jRes;
}

void PoolManager::start()
{
    m_running.store(true, std::memory_order_relaxed);
    m_async_pending.store(true, std::memory_order_relaxed);
    m_connectionSwitches.fetch_add(1, std::memory_order_relaxed);
    g_io_service.post(m_io_strand.wrap(boost::bind(&PoolManager::rotateConnect, this)));
}

void PoolManager::rotateConnect()
{
    if (p_client && p_client->isConnected())
        return;

    // Check we're within bounds
    if (m_activeConnectionIdx >= m_Settings.connections.size())
        m_activeConnectionIdx = 0;

    // If this connection is marked Unrecoverable then discard it
    if (m_Settings.connections.at(m_activeConnectionIdx)->IsUnrecoverable())
    {
        m_Settings.connections.erase(m_Settings.connections.begin() + m_activeConnectionIdx);
        m_connectionAttempt = 0;
        if (m_activeConnectionIdx >= m_Settings.connections.size())
            m_activeConnectionIdx = 0;
        m_connectionSwitches.fetch_add(1, std::memory_order_relaxed);
    }
    else if (m_connectionAttempt >= m_Settings.connectionMaxRetries)
    {
        // If this is the only connection we can't rotate
        // forever
        if (m_Settings.connections.size() == 1)
        {
            m_Settings.connections.erase(m_Settings.connections.begin() + m_activeConnectionIdx);
        }
        // Rotate connections if above max attempts threshold
        else
        {
            m_connectionAttempt = 0;
            m_activeConnectionIdx++;
            if (m_activeConnectionIdx >= m_Settings.connections.size())
                m_activeConnectionIdx = 0;
            m_connectionSwitches.fetch_add(1, std::memory_order_relaxed);
        }
    }

    if (!m_Settings.connections.empty() && m_Settings.connections.at(m_activeConnectionIdx)->Host() != "exit")
    {
        if (p_client)
            p_client = nullptr;

        if (m_Settings.connections.at(m_activeConnectionIdx)->Family() == ProtocolFamily::GETWORK)
            p_client =
                std::unique_ptr<PoolClient>(new EthGetworkClient(m_Settings.noWorkTimeout, m_Settings.getWorkPollInterval));
        if (m_Settings.connections.at(m_activeConnectionIdx)->Family() == ProtocolFamily::STRATUM)
            p_client = std::unique_ptr<PoolClient>(
                new EthStratumClient(m_Settings.noWorkTimeout, m_Settings.noResponseTimeout));
        if (m_Settings.connections.at(m_activeConnectionIdx)->Family() == ProtocolFamily::SIMULATION)
            p_client = std::unique_ptr<PoolClient>(new SimulateClient(m_Settings.benchmarkBlock));

        if (p_client)
            setClientHandlers();

        // Count connectionAttempts
        m_connectionAttempt++;

        // Invoke connections
        m_selectedHost = m_Settings.connections.at(m_activeConnectionIdx)->Host() + ":" +
                         to_string(m_Settings.connections.at(m_activeConnectionIdx)->Port());
        p_client->setConnection(m_Settings.connections.at(m_activeConnectionIdx));
        cnote << "Selected pool " << m_selectedHost;
 
        
        if ((m_connectionAttempt > 1) && (m_Settings.delayBeforeRetry > 0))
        {
            cnote << "Next connection attempt in " << m_Settings.delayBeforeRetry << " seconds";
            m_reconnecttimer.expires_from_now(boost::posix_time::seconds(m_Settings.delayBeforeRetry));
            m_reconnecttimer.async_wait(m_io_strand.wrap(boost::bind(
                &PoolManager::reconnecttimer_elapsed, this, boost::asio::placeholders::error)));
        }
        else
        {
            p_client->connect();
        }
    }
    else
    {

        if (m_Settings.connections.empty())
            cnote << "No more connections to try. Exiting...";
        else
            cnote << "'exit' failover just got hit. Exiting...";

        // Stop mining if applicable
        if (Farm::f().isMining())
        {
            cnote << "Shutting down miners...";
            Farm::f().stop();
        }

        m_running.store(false, std::memory_order_relaxed);
        raise(SIGTERM);
    }
}

void PoolManager::showMiningAt()
{
    // Should not happen
    if (!m_currentWp)
        return;

    double d = dev::getHashesToTarget(m_currentWp.boundary.hex(HexPrefix::Add));
    cnote << "Epoch : " EthWhite << m_currentWp.epoch << EthReset << " Difficulty : " EthWhite
          << dev::getFormattedHashes(d) << EthReset;
}

void PoolManager::failovertimer_elapsed(const boost::system::error_code& ec)
{
    if (!ec)
    {
        if (m_running.load(std::memory_order_relaxed))
        {
            if (m_activeConnectionIdx != 0)
            {
                m_activeConnectionIdx = 0;
                m_connectionAttempt = 0;
                m_connectionSwitches.fetch_add(1, std::memory_order_relaxed);
                cnote << "Failover timeout reached, retrying connection to primary pool";
                p_client->disconnect();
            }
        }
    }
}

void PoolManager::submithrtimer_elapsed(const boost::system::error_code& ec)
{
    if (!ec)
    {
        if (m_running.load(std::memory_order_relaxed))
        {

            if (p_client && p_client->isConnected())
                p_client->submitHashrate((uint32_t)Farm::f().HashRate(), m_Settings.hashRateId);

            // Resubmit actor
            m_submithrtimer.expires_from_now(boost::posix_time::seconds(m_Settings.hashRateInterval));
            m_submithrtimer.async_wait(m_io_strand.wrap(boost::bind(
                &PoolManager::submithrtimer_elapsed, this, boost::asio::placeholders::error)));
        }
    }
}

void PoolManager::reconnecttimer_elapsed(const boost::system::error_code& ec)
{
    if (ec)
        return;

    if (m_running.load(std::memory_order_relaxed))
    {
        if (p_client && !p_client->isConnected())
        {
            p_client->connect();
        }
    }
}

int PoolManager::getCurrentEpoch()
{
    return m_currentWp.epoch;
}

double PoolManager::getCurrentDifficulty()
{
    if (!m_currentWp)
        return 0.0;

    return dev::getHashesToTarget(m_currentWp.boundary.hex(HexPrefix::Add));
}

unsigned PoolManager::getConnectionSwitches()
{
    return m_connectionSwitches.load(std::memory_order_relaxed);
}

unsigned PoolManager::getEpochChanges()
{
<<<<<<< HEAD
    return m_epochChanges.load(std::memory_order_relaxed);
=======
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

		cnote << "Selected pool" << (m_connections[m_activeConnectionIdx].Host() + ":" + toString(m_connections[m_activeConnectionIdx].Port()));
		p_client->connect();
		return;
	}

	// Fallback logic, tries current connection multiple times and then switches to
	// one of the other connections.
	if (m_reconnectTries > m_reconnectTry) {

		m_reconnectTry++;
		cnote << "Selected pool" << (m_connections[m_activeConnectionIdx].Host() + ":" + toString(m_connections[m_activeConnectionIdx].Port()));
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
			cnote << "Selected pool" << (m_connections[m_activeConnectionIdx].Host() + ":" + toString(m_connections[m_activeConnectionIdx].Port()));
			p_client->connect();
		}
	}
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
}
