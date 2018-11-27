#pragma once

#include <iostream>

#include <json/json.h>

#include <libdevcore/Worker.h>
#include <libethcore/Farm.h>
#include <libethcore/Miner.h>

#include "PoolClient.h"
#include "getwork/EthGetworkClient.h"
#include "stratum/EthStratumClient.h"
#include "testing/SimulateClient.h"

using namespace std;

namespace dev
{
namespace eth
{
class PoolManager
{
public:
    PoolManager(unsigned maxTries, unsigned failovertimeout, unsigned ergodicity,
        bool reportHashrate, unsigned workTimeout, unsigned responseTimeout, unsigned pollInterval,
        unsigned benchmarkBlock);
    static PoolManager& p() { return *m_this; }
    void addConnection(URI& conn);
    void clearConnections();
    Json::Value getConnectionsJson();
    int setActiveConnection(unsigned int idx);
    URI getActiveConnectionCopy();
    int removeConnection(unsigned int idx);
    void start();
    void stop();
    bool isConnected() { return p_client->isConnected(); };
    bool isRunning() { return m_running; };
    int getCurrentEpoch();
    double getCurrentDifficulty();
    unsigned getConnectionSwitches();
    unsigned getEpochChanges();

private:
    void rotateConnect();

    void setClientHandlers();
    void unsetClientHandlers();

    void showEpoch();
    void showDifficulty();

    unsigned m_hrReportingInterval = 60;

    unsigned m_failoverTimeout;  // After this amount of time in minutes of mining on a failover
                                 // pool return to "primary"

    unsigned m_workTimeout;  // Amount of time, in seconds, with no work which causes a
                             // disconnection

    unsigned m_responseTimeout;  // Amount of time, in milliseconds, with no response from pool
                                 // which causes a disconnection

    unsigned m_pollInterval;  // Interval, in milliseconds, among polls to a getwork provider

    unsigned m_benchmarkBlock;  // Block number to test simulation against

    void failovertimer_elapsed(const boost::system::error_code& ec);
    void submithrtimer_elapsed(const boost::system::error_code& ec);

    std::atomic<bool> m_running = {false};
    std::atomic<bool> m_stopping = {false};

    bool m_hashrate;           // Whether or not submit hashrate to work provider (pool)
    std::string m_hashrateId;  // The unique client Id to use when submitting hashrate
    unsigned m_ergodicity = 0;
    unsigned m_connectionAttempt = 0;
    unsigned m_maxConnectionAttempts = 0;
    std::string m_selectedHost = "";  // Holds host name (and endpoint) of selected connection
    std::atomic<unsigned> m_connectionSwitches = {0};

    std::vector<URI> m_connections;
    unsigned m_activeConnectionIdx = 0;
    mutable Mutex m_activeConnectionMutex;

    WorkPackage m_currentWp;

    boost::asio::io_service::strand m_io_strand;
    boost::asio::deadline_timer m_failovertimer;
    boost::asio::deadline_timer m_submithrtimer;

    PoolClient* p_client = nullptr;

    std::atomic<unsigned> m_epochChanges = {0};

    static PoolManager* m_this;
};

}  // namespace eth
}  // namespace dev
