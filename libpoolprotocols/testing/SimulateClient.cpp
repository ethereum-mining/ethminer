#include <chrono>

#include "SimulateClient.h"

using namespace std;
using namespace std::chrono;
using namespace dev;
using namespace eth;

SimulateClient::SimulateClient(unsigned const& difficulty, unsigned const& block)
  : PoolClient(), Worker("simulator")
{
    m_difficulty = difficulty - 1;
    m_block = block;
}

SimulateClient::~SimulateClient() {}

void SimulateClient::connect()
{
    m_connected.store(true, std::memory_order_relaxed);
    m_uppDifficulty = true;

    if (m_onConnected)
    {
        m_onConnected();
    }

    // No need to worry about starting again.
    // Worker class prevents that
    startWorking();
}

void SimulateClient::disconnect()
{
    m_connected.store(false, std::memory_order_relaxed);

    if (m_onDisconnected)
    {
        m_onDisconnected();
    }
}

void SimulateClient::submitHashrate(string const& rate)
{
    (void)rate;
    auto sec = duration_cast<seconds>(steady_clock::now() - m_time);
    cnote << "On difficulty" << m_difficulty << "for" << sec.count() << "seconds";
}

void SimulateClient::submitSolution(const Solution& solution)
{
    m_uppDifficulty = true;
    cnote << "Difficulty:" << m_difficulty;
    std::chrono::steady_clock::time_point submit_start = std::chrono::steady_clock::now();
    bool accepted =
        EthashAux::eval(solution.work.epoch, solution.work.header, solution.nonce).value <
        solution.work.boundary;
    std::chrono::milliseconds response_delay_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - submit_start);

    if (accepted)
    {
        if (m_onSolutionAccepted)
        {
            m_onSolutionAccepted(false, response_delay_ms, solution.index);
        }
    }
    else
    {
        if (m_onSolutionRejected)
        {
            m_onSolutionRejected(false, response_delay_ms, solution.index);
        }
    }
}

// Handles all logic here
void SimulateClient::workLoop()
{
    cout << "Preparing DAG for block #" << m_block << endl;
    BlockHeader genesis;
    genesis.setNumber(m_block);
    WorkPackage current = WorkPackage(genesis);
    m_time = std::chrono::steady_clock::now();
    while (true)
    {
        if (m_connected.load(std::memory_order_relaxed))
        {
            if (m_uppDifficulty)
            {
                m_uppDifficulty = false;

                auto sec = duration_cast<seconds>(steady_clock::now() - m_time);
                cnote << "Took" << sec.count() << "seconds at" << m_difficulty
                      << "difficulty to find solution";

                if (sec.count() < 12)
                {
                    m_difficulty++;
                }
                if (sec.count() > 18)
                {
                    m_difficulty--;
                }

                cnote << "Now using difficulty " << m_difficulty;
                m_time = std::chrono::steady_clock::now();
                if (m_onWorkReceived)
                {
                    genesis.setDifficulty(u256(1) << m_difficulty);
                    genesis.noteDirty();

                    current.header = h256::random();
                    current.boundary = genesis.boundary();

                    m_onWorkReceived(current);
                }
            }
            else
            {
                this_thread::sleep_for(chrono::milliseconds(100));
            }
        }
        else
        {
            this_thread::sleep_for(chrono::seconds(5));
        }
    }
}
