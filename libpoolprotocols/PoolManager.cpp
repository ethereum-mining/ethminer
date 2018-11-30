#include <chrono>

#include "PoolManager.h"

using namespace std;
using namespace dev;
using namespace eth;

PoolManager* PoolManager::m_this = nullptr;

PoolManager::PoolManager(unsigned maxTries, unsigned failoverTimeout, unsigned ergodicity,
    bool reportHashrate, unsigned workTimeout, unsigned responseTimeout, unsigned pollInterval, unsigned benchmarkBlock)
  : m_hashrate(reportHashrate),
    m_io_strand(g_io_service),
    m_failovertimer(g_io_service),
    m_submithrtimer(g_io_service)
{
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "PoolManager::PoolManager() begin");

    m_this = this;
    m_ergodicity = ergodicity;
    m_maxConnectionAttempts = maxTries;
    m_failoverTimeout = failoverTimeout;
    m_workTimeout = workTimeout;
    m_responseTimeout = responseTimeout;
    m_pollInterval = pollInterval;
    m_benchmarkBlock = benchmarkBlock;

    m_currentWp.header = h256();

    // If hashrate submission required compute a random
    // unique id
    if (m_hashrate)
        m_hashrateId = "0x" + h256::random().hex();

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
            cnote << string(EthRed "Solution 0x") + toHex(sol.nonce)
                  << " wasted. Waiting for connection...";
        }

        return false;
    });


    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "PoolManager::PoolManager() end");
}

void PoolManager::setClientHandlers() {

    p_client->onConnected([&]() {
        {
            Guard l(m_activeConnectionMutex);

            // If HostName is already an IP address no need to append the
            // effective ip address.
            if (p_client->getConnection()->HostNameType() == dev::UriHostNameType::Dns ||
                p_client->getConnection()->HostNameType() == dev::UriHostNameType::Basic)
                m_selectedHost.append(p_client->ActiveEndPoint());

            cnote << "Established connection to " << m_selectedHost;

            // Reset current WorkPackage
            m_currentWp.job.clear();
            m_currentWp.header = h256();

            // Shuffle if needed
            if (m_ergodicity == 1)
                Farm::f().shuffle();

            // Rough implementation to return to primary pool
            // after specified amount of time
            if (m_activeConnectionIdx != 0 && m_failoverTimeout > 0)
            {
                m_failovertimer.expires_from_now(boost::posix_time::minutes(m_failoverTimeout));
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
        if (m_hashrate)
        {
            m_submithrtimer.expires_from_now(boost::posix_time::seconds(m_hrReportingInterval));
            m_submithrtimer.async_wait(m_io_strand.wrap(boost::bind(
                &PoolManager::submithrtimer_elapsed, this, boost::asio::placeholders::error)));
        }
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
        bool newEpoch = (_currentEpoch == -1 || wp.seed != m_currentWp.seed);
        bool newDiff = (wp.boundary != m_currentWp.boundary);
        m_currentWp = wp;

        if (newEpoch)
        {
            m_epochChanges.fetch_add(1, std::memory_order_relaxed);
            if (m_currentWp.block > 0)
                m_currentWp.epoch = m_currentWp.block / 30000;
            else
                m_currentWp.epoch =
                    ethash::find_epoch_number(ethash::hash256_from_bytes(m_currentWp.seed.data()));
        }
        else
        {
            m_currentWp.epoch = _currentEpoch;
        }

        if (newDiff || newEpoch)
            showMiningAt();


        cnote << "Job: " EthWhite "#" << m_currentWp.header.abridged()
              << (m_currentWp.block != -1 ? (" block " + to_string(m_currentWp.block)) : "")
              << EthReset << " " << m_selectedHost;

        // Shuffle if needed
        if (m_ergodicity == 2 && m_currentWp.exSizeBytes == 0)
            Farm::f().shuffle();

        Farm::f().setWork(m_currentWp);
    });

    p_client->onSolutionAccepted(
        [&](std::chrono::milliseconds const& elapsedMs, unsigned const& miner_index) {
            std::stringstream ss;
            ss << std::setw(4) << std::setfill(' ') << elapsedMs.count() << " ms."
               << " " << m_selectedHost;
            cnote << EthLime "**Accepted" EthReset << ss.str();
            Farm::f().acceptedSolution(miner_index);
        });

    p_client->onSolutionRejected(
        [&](std::chrono::milliseconds const& elapsedMs, unsigned const& miner_index) {
            std::stringstream ss;
            ss << std::setw(4) << std::setfill(' ') << elapsedMs.count() << "ms."
               << "   " << m_selectedHost;
            cwarn << EthRed "**Rejected" EthReset << ss.str();
            Farm::f().rejectedSolution(miner_index);
        });

}

void PoolManager::stop()
{
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "PoolManager::stop() begin");
    if (m_running.load(std::memory_order_relaxed))
    {
        m_stopping.store(true, std::memory_order_relaxed);

        if (p_client && p_client->isConnected())
        {
            p_client->disconnect();
            // Wait for async operations to complete
            while (m_running.load(std::memory_order_relaxed))
                this_thread::sleep_for(chrono::milliseconds(500));

            delete p_client;
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

void PoolManager::addConnection(URI& conn)
{
    Guard l(m_activeConnectionMutex);
    m_connections.push_back(conn);
}

/*
 * Remove a connection
 * Returns:  0 on success
 *          -1 failure (out of bounds)
 *          -2 failure (active connection should be deleted)
 */
int PoolManager::removeConnection(unsigned int idx)
{
    Guard l(m_activeConnectionMutex);
    if (idx >= m_connections.size())
        return -1;
    if (idx == m_activeConnectionIdx)
        return -2;
    m_connections.erase(m_connections.begin() + idx);
    if (m_activeConnectionIdx > idx)
    {
        m_activeConnectionIdx--;
    }
    return 0;
}

void PoolManager::clearConnections()
{
    {
        Guard l(m_activeConnectionMutex);
        m_connections.clear();
    }
    if (p_client && p_client->isConnected())
        p_client->disconnect();
}

int PoolManager::setActiveConnectionCommon(unsigned int idx, UniqueGuard& l)
{
    if (idx == m_activeConnectionIdx)
        return 0;

    m_connectionSwitches.fetch_add(1, std::memory_order_relaxed);
    m_activeConnectionIdx = idx;
    m_connectionAttempt = 0;
    l.unlock();
    p_client->disconnect();

    return 0;
}

/*
 * Sets the active connection
 * Returns: 0 on success, -1 on failure (out of bounds)
 */
int PoolManager::setActiveConnection(unsigned int idx)
{
    // Sets the active connection to the requested index
    UniqueGuard l(m_activeConnectionMutex);
    if (idx >= m_connections.size())
        return -1;
    return setActiveConnectionCommon(idx, l);
}

int PoolManager::setActiveConnection(std::string& host)
{
    std::regex r(host);
    UniqueGuard l(m_activeConnectionMutex);
    for (size_t idx = 0; idx < m_connections.size(); idx++)
        if (std::regex_match(m_connections[idx].str(), r))
            return setActiveConnectionCommon(idx, l);
    return -1;
}

URI PoolManager::getActiveConnectionCopy()
{
    Guard l(m_activeConnectionMutex);
    if (m_connections.size() > m_activeConnectionIdx)
        return m_connections[m_activeConnectionIdx];
    return URI(":0");
}

Json::Value PoolManager::getConnectionsJson()
{
    // Returns the list of configured connections
    Json::Value jRes;
    Guard l(m_activeConnectionMutex);

    for (size_t i = 0; i < m_connections.size(); i++)
    {
        Json::Value JConn;
        JConn["index"] = (unsigned)i;
        JConn["active"] = (i == m_activeConnectionIdx ? true : false);
        JConn["uri"] = m_connections[i].str();
        jRes.append(JConn);
    }

    return jRes;
}

void PoolManager::start()
{
    Guard l(m_activeConnectionMutex);
    m_running.store(true, std::memory_order_relaxed);
    m_connectionSwitches.fetch_add(1, std::memory_order_relaxed);
    g_io_service.post(m_io_strand.wrap(boost::bind(&PoolManager::rotateConnect, this)));
}

void PoolManager::rotateConnect()
{
    if (p_client && p_client->isConnected())
        return;

    UniqueGuard l(m_activeConnectionMutex);

    // Check we're within bounds
    if (m_activeConnectionIdx >= m_connections.size())
        m_activeConnectionIdx = 0;

    // If this connection is marked Unrecoverable then discard it
    if (m_connections.at(m_activeConnectionIdx).IsUnrecoverable())
    {
        m_connections.erase(m_connections.begin() + m_activeConnectionIdx);
        m_connectionAttempt = 0;
        if (m_activeConnectionIdx >= m_connections.size())
            m_activeConnectionIdx = 0;
        m_connectionSwitches.fetch_add(1, std::memory_order_relaxed);
    }
    else if (m_connectionAttempt >= m_maxConnectionAttempts)
    {
        // If this is the only connection we can't rotate
        // forever
        if (m_connections.size() == 1)
        {
            m_connections.erase(m_connections.begin() + m_activeConnectionIdx);
        }
        // Rotate connections if above max attempts threshold
        else
        {
            m_connectionAttempt = 0;
            m_activeConnectionIdx++;
            if (m_activeConnectionIdx >= m_connections.size())
                m_activeConnectionIdx = 0;
            m_connectionSwitches.fetch_add(1, std::memory_order_relaxed);
        }
    }

    if (!m_connections.empty() && m_connections.at(m_activeConnectionIdx).Host() != "exit")
    {
        if (p_client) delete p_client;

        if (m_connections.at(m_activeConnectionIdx).Family() == ProtocolFamily::GETWORK)
            p_client = new EthGetworkClient(m_workTimeout, m_pollInterval);
        if (m_connections.at(m_activeConnectionIdx).Family() == ProtocolFamily::STRATUM)
            p_client = new EthStratumClient(m_workTimeout, m_responseTimeout);
        if (m_connections.at(m_activeConnectionIdx).Family() == ProtocolFamily::SIMULATION)
            p_client = new SimulateClient(m_benchmarkBlock);

        if (p_client)
            setClientHandlers();
        
        // Count connectionAttempts
        m_connectionAttempt++;

        // Invoke connections
        m_selectedHost = m_connections.at(m_activeConnectionIdx).Host() + ":" +
                         to_string(m_connections.at(m_activeConnectionIdx).Port());
        p_client->setConnection(&m_connections.at(m_activeConnectionIdx));
        cnote << "Selected pool " << m_selectedHost;

        l.unlock();
        p_client->connect();
    }
    else
    {
        l.unlock();

        if (m_connections.empty())
        {
            cnote << "No more connections to try. Exiting...";
        }
        else
        {
            cnote << "'exit' failover just got hit. Exiting...";
        }

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

    static const char* suffixes[] = {"h", "Kh", "Mh", "Gh"};
    double d = getCurrentDifficulty();
    unsigned i;

    for (i = 0; i < 3; i++)
    {
        if (d < 1000.0)
            break;
        d /= 1000.0;
    }

    std::stringstream ss;
    ss << fixed << setprecision(2) << d << " " << suffixes[i];
    cnote << "Epoch : " EthWhite << m_currentWp.epoch << EthReset << " Difficulty : " EthWhite
          << ss.str() << EthReset;
}

void PoolManager::failovertimer_elapsed(const boost::system::error_code& ec)
{
    if (!ec)
    {
        if (m_running.load(std::memory_order_relaxed))
        {
            UniqueGuard l(m_activeConnectionMutex);
            if (m_activeConnectionIdx != 0)
            {
                m_activeConnectionIdx = 0;
                m_connectionAttempt = 0;
                m_connectionSwitches.fetch_add(1, std::memory_order_relaxed);
                l.unlock();
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
            {
                auto mp = Farm::f().miningProgress();
                std::string h = toHex(toCompactBigEndian(uint64_t(mp.hashRate), 1));
                std::string res = h[0] != '0' ? h : h.substr(1);

                // Should be 32 bytes
                // https://github.com/ethereum/wiki/wiki/JSON-RPC#eth_submithashrate
                std::ostringstream ss;
                ss << "0x" << std::setw(64) << std::setfill('0') << res;
                p_client->submitHashrate(ss.str(), m_hashrateId);
            }

            // Resubmit actor
            m_submithrtimer.expires_from_now(boost::posix_time::seconds(m_hrReportingInterval));
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

    using namespace boost::multiprecision;
    static const uint256_t dividend(
        "0xffff000000000000000000000000000000000000000000000000000000000000");
    const uint256_t divisor(string("0x") + m_currentWp.boundary.hex());
    std::stringstream ss;
    return double(dividend / divisor);
}

unsigned PoolManager::getConnectionSwitches()
{
    return m_connectionSwitches.load(std::memory_order_relaxed);
}

unsigned PoolManager::getEpochChanges()
{
    return m_epochChanges.load(std::memory_order_relaxed);
}
