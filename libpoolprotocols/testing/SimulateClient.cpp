#include <chrono>

#include "SimulateClient.h"

using namespace std;
using namespace std::chrono;
using namespace dev;
using namespace eth;

SimulateClient::SimulateClient(unsigned const& block)
  : PoolClient(), Worker("sim")
{
    m_block = block;
}

SimulateClient::~SimulateClient() = default;

void SimulateClient::connect()
{
    m_connected.store(true, std::memory_order_relaxed);

    if (m_onConnected)
        m_onConnected();

    // No need to worry about starting again.
    // Worker class prevents that
    startWorking();
}

void SimulateClient::disconnect()
{
    m_connected.store(false, std::memory_order_relaxed);
    if (m_onDisconnected)
        m_onDisconnected();
}

void SimulateClient::submitHashrate(string const& rate, string const& id)
{
    (void)rate;
    (void)id;
}

void SimulateClient::submitSolution(const Solution& solution)
{
    // This is a fake submission only evaluated locally
    std::chrono::steady_clock::time_point submit_start = std::chrono::steady_clock::now();
    bool accepted =
        EthashAux::eval(solution.work.epoch, solution.work.header, solution.nonce).value <=
        solution.work.boundary;
    std::chrono::milliseconds response_delay_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - submit_start);

    if (accepted)
    {
        if (m_onSolutionAccepted)
            m_onSolutionAccepted(response_delay_ms, solution.midx);
    }
    else
    {
        if (m_onSolutionRejected)
            m_onSolutionRejected(response_delay_ms, solution.midx);
    }
        
}

// Handles all logic here
void SimulateClient::workLoop()
{

    m_start_time = std::chrono::steady_clock::now();

    WorkPackage current;
    current.seed = h256::random();  // We don't actually need a real seed as the epoch
                                    // is calculated upon block number (see poolmanager)
    current.header = h256::random();
    current.block = m_block;
    current.boundary = h256(dev::getTargetFromDiff(1));
    m_onWorkReceived(current);  // submit new fake job

    while (m_connected.load(std::memory_order_relaxed))
    {

        current.header = h256::random();
        m_onWorkReceived(current);

        this_thread::sleep_for(chrono::seconds(15));  // average block time
    }
}
