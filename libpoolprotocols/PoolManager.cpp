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
    m_submithrtimer(g_io_service)
{
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
}

void PoolManager::setClientHandlers()
{
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
                if (m_currentWp.block > 0)
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
    bool found = false;
    for (size_t idx = 0; idx < m_Settings.connections.size(); idx++)
        if (boost::iequals(m_Settings.connections[idx]->str(), _connstring))
        {
            found = true;
            setActiveConnectionCommon(idx);
            break;
        }
    if (!found)
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

        p_client->connect();
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
    return m_epochChanges.load(std::memory_order_relaxed);
}
