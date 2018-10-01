#include <chrono>

#include "PoolManager.h"

using namespace std;
using namespace dev;
using namespace eth;

PoolManager* PoolManager::m_this = nullptr;

PoolManager::PoolManager(
    PoolClient* client, MinerType const& minerType, unsigned maxTries, unsigned failoverTimeout)
  : m_io_strand(g_io_service), m_failovertimer(g_io_service), m_minerType(minerType)
{
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "PoolManager::PoolManager() begin");

    m_this = this;
    p_client = client;
    m_maxConnectionAttempts = maxTries;
    m_failoverTimeout = failoverTimeout;

    p_client->onConnected([&]() {
        {
            Guard l(m_activeConnectionMutex);
            m_lastConnectedHost = m_connections.at(m_activeConnectionIdx).Host();
            cnote << "Established connection with "
                  << (m_lastConnectedHost + ":" +
                         toString(m_connections.at(m_activeConnectionIdx).Port()))
                  << " at " << p_client->ActiveEndPoint();

            // Rough implementation to return to primary pool
            // after specified amount of time
            if (m_activeConnectionIdx != 0 && m_failoverTimeout > 0)
            {
                m_failovertimer.expires_from_now(boost::posix_time::minutes(m_failoverTimeout));
                m_failovertimer.async_wait(m_io_strand.wrap(boost::bind(
                    &PoolManager::check_failover_timeout, this, boost::asio::placeholders::error)));
            }
            else
            {
                m_failovertimer.cancel();
            }
        }

        if (!Farm::f().isMining())
        {
            cnote << "Spinning up miners...";
            if (m_minerType == MinerType::CL)
                Farm::f().start("opencl", false);
            else if (m_minerType == MinerType::CUDA)
                Farm::f().start("cuda", false);
            else if (m_minerType == MinerType::Mixed)
            {
                Farm::f().start("cuda", false);
                Farm::f().start("opencl", true);
            }
        }
    });

    p_client->onDisconnected([&]() {
        dev::setThreadName("main");
        cnote << "Disconnected from " + m_lastConnectedHost << p_client->ActiveEndPoint();

        // Do not stop mining here
        // Workloop will determine if we're trying a fast reconnect to same pool
        // or if we're switching to failover(s)
    });

    p_client->onWorkReceived([&](WorkPackage const& wp) {

        cnote << "Job: " EthWhite "#" << wp.header.abridged() << EthReset " "
              << m_lastConnectedHost << p_client->ActiveEndPoint();
        if (wp.boundary != m_lastBoundary)
        {
            using namespace boost::multiprecision;

            m_lastBoundary = wp.boundary;
            static const uint256_t dividend(
                "0xffff000000000000000000000000000000000000000000000000000000000000");
            const uint256_t divisor(string("0x") + m_lastBoundary.hex());
            std::stringstream ss;
            m_lastDifficulty = double(dividend / divisor);
            ss << fixed << setprecision(2) << m_lastDifficulty / 1000000000.0
               << "K megahash";
            cnote << "Pool difficulty: " EthWhite << ss.str() << EthReset;
        }
        if (wp.epoch != m_lastEpoch)
        {
            cnote << "New epoch " EthWhite << wp.epoch << EthReset;
            m_lastEpoch = wp.epoch;
            m_epochChanges.fetch_add(1, std::memory_order_relaxed);
        }

        Farm::f().setWork(wp);
    });

    p_client->onSolutionAccepted([&](bool const& stale,
                                     std::chrono::milliseconds const& elapsedMs, unsigned const& miner_index) {

        std::stringstream ss;
        ss << std::setw(4) << std::setfill(' ') << elapsedMs.count() << " ms."
           << " " << m_lastConnectedHost + p_client->ActiveEndPoint();
        cnote << EthLime "**Accepted" EthReset << (stale ? EthYellow "(stale)" EthReset : "")
              << ss.str();
        Farm::f().acceptedSolution(stale, miner_index);
    });

    p_client->onSolutionRejected([&](bool const& stale,
                                     std::chrono::milliseconds const& elapsedMs, unsigned const& miner_index) {

        std::stringstream ss;
        ss << std::setw(4) << std::setfill(' ') << elapsedMs.count() << "ms."
           << "   " << m_lastConnectedHost + p_client->ActiveEndPoint();
        cwarn << EthRed "**Rejected" EthReset << (stale ? EthYellow "(stale)" EthReset : "")
              << ss.str();
        Farm::f().rejectedSolution(miner_index);
    });

    Farm::f().onSolutionFound([&](const Solution& sol) {
        // Solution should passthrough only if client is
        // properly connected. Otherwise we'll have the bad behavior
        // to log nonce submission but receive no response

        if (p_client->isConnected())
        {

            if (sol.stale)
                cwarn << "Stale solution: " << EthWhite "0x" << toHex(sol.nonce) << EthReset;
            else
                cnote << "Solution: " << EthWhite "0x" << toHex(sol.nonce) << EthReset;

            p_client->submitSolution(sol);
        }
        else
        {
            cnote << string(EthRed "Solution 0x") + toHex(sol.nonce)
                  << " wasted. Waiting for connection...";
        }

        return false;
    });
    Farm::f().onMinerRestart([&]() {
        dev::setThreadName("main");
        cnote << "Restart miners...";

        if (Farm::f().isMining())
        {
            cnote << "Shutting down miners...";
            Farm::f().stop();
        }

        cnote << "Spinning up miners...";
        if (m_minerType == MinerType::CL)
            Farm::f().start("opencl", false);
        else if (m_minerType == MinerType::CUDA)
            Farm::f().start("cuda", false);
        else if (m_minerType == MinerType::Mixed)
        {
            Farm::f().start("cuda", false);
            Farm::f().start("opencl", true);
        }
    });

    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "PoolManager::PoolManager() end");
}

void PoolManager::stop()
{
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "PoolManager::stop() begin");

    if (m_running.load(std::memory_order_relaxed))
    {
        cnote << "Shutting down...";

        m_running.store(false, std::memory_order_relaxed);
        m_failovertimer.cancel();

        if (p_client->isConnected())
            p_client->disconnect();

        if (Farm::f().isMining())
        {
            cnote << "Shutting down miners...";
            Farm::f().stop();
        }
    }

    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "PoolManager::stop() end");
}

void PoolManager::workLoop()
{
    dev::setThreadName("main");

    while (m_running.load(std::memory_order_relaxed))
    {
        // Take action only if not pending state (connecting/disconnecting)
        // Otherwise do nothing and wait until connection state is NOT pending
        if (!p_client->isPendingState())
        {
            if (!p_client->isConnected())
            {
                // As we're not connected: suspend mining if we're still searching a solution
                suspendMining();

                UniqueGuard l(m_activeConnectionMutex);

                // If this connection is marked Unrecoverable then discard it
                if (m_connections.at(m_activeConnectionIdx).IsUnrecoverable())
                {
                    p_client->unsetConnection();

                    m_connections.erase(m_connections.begin() + m_activeConnectionIdx);

                    m_connectionAttempt = 0;
                    if (m_activeConnectionIdx >= m_connections.size())
                    {
                        m_activeConnectionIdx = 0;
                    }
                    m_connectionSwitches.fetch_add(1, std::memory_order_relaxed);
                }
                else if (m_connectionAttempt >= m_maxConnectionAttempts)
                {
                    // Rotate connections if above max attempts threshold
                    m_connectionAttempt = 0;
                    m_activeConnectionIdx++;
                    if (m_activeConnectionIdx >= m_connections.size())
                    {
                        m_activeConnectionIdx = 0;
                    }
                    m_connectionSwitches.fetch_add(1, std::memory_order_relaxed);
                }

                if (!m_connections.empty() &&
                    m_connections.at(m_activeConnectionIdx).Host() != "exit")
                {
                    // Count connectionAttempts
                    m_connectionAttempt++;

                    // Invoke connections
                    p_client->setConnection(&m_connections.at(m_activeConnectionIdx));
                    cnote << "Selected pool "
                          << (m_connections.at(m_activeConnectionIdx).Host() + ":" +
                                 toString(m_connections.at(m_activeConnectionIdx).Port()));

                    l.unlock();

                    // Clean any list of jobs inherited from
                    // previous connection
                    p_client->connect();
                }
                else
                {
                    l.unlock();

                    cnote << "No more connections to try. Exiting...";

                    // Stop mining if applicable
                    if (Farm::f().isMining())
                    {
                        cnote << "Shutting down miners...";
                        Farm::f().stop();
                    }

                    m_running.store(false, std::memory_order_relaxed);
                    continue;
                }
            }
        }

        // Hashrate reporting
        m_hashrateReportingTimePassed++;

        if (m_hashrateReportingTimePassed > m_hashrateReportingTime)
        {
            auto mp = Farm::f().miningProgress();
            std::string h = toHex(toCompactBigEndian(uint64_t(mp.hashRate), 1));
            std::string res = h[0] != '0' ? h : h.substr(1);

            // Should be 32 bytes
            // https://github.com/ethereum/wiki/wiki/JSON-RPC#eth_submithashrate
            std::ostringstream ss;
            ss << std::setw(64) << std::setfill('0') << res;

            p_client->submitHashrate("0x" + ss.str());
            m_hashrateReportingTimePassed = 0;
        }

        this_thread::sleep_for(chrono::seconds(1));
    }
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
    if (idx == m_activeConnectionIdx)
        return 0;

    m_connectionSwitches.fetch_add(1, std::memory_order_relaxed);
    m_activeConnectionIdx = idx;
    m_connectionAttempt = 0;
    l.unlock();
    p_client->disconnect();

    // Suspend mining if applicable as we're switching
    suspendMining();
    return 0;
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
        JConn["uri"] = m_connections[i].String();
        jRes.append(JConn);
    }

    return jRes;
}

void PoolManager::start()
{
    Guard l(m_activeConnectionMutex);
    if (m_connections.size() > 0)
    {
        m_running.store(true, std::memory_order_relaxed);
        m_workThread = std::thread{boost::bind(&PoolManager::workLoop, this)};
    }
    else
    {
        cwarn << "Manager has no connections defined!";
    }
}

void PoolManager::check_failover_timeout(const boost::system::error_code& ec)
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

void PoolManager::suspendMining()
{
    if (!Farm::f().isMining())
        return;

    WorkPackage wp = Farm::f().work();
    if (!wp)
        return;

    Farm::f().setWork({}); /* suspend by setting empty work package */
    cnote << "Suspend mining due connection change...";
}

double PoolManager::getCurrentDifficulty()
{
    if (!m_running.load(std::memory_order_relaxed))
        return 0.0;
    if (!p_client->isConnected())
        return 0.0;
    return m_lastDifficulty;
}

unsigned PoolManager::getConnectionSwitches()
{
    return m_connectionSwitches.load(std::memory_order_relaxed);
}

unsigned PoolManager::getEpochChanges()
{
    return m_epochChanges.load(std::memory_order_relaxed);
}
