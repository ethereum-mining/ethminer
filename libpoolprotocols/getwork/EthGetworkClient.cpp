#include "EthGetworkClient.h"

#include <chrono>

#include <ethash/ethash.hpp>

using namespace std;
using namespace dev;
using namespace eth;

EthGetworkClient::EthGetworkClient(unsigned farmRecheckPeriod, bool submitHashrate)
  : PoolClient(), Worker("getwork"), m_submit_hashrate(submitHashrate)
{
    m_farmRecheckPeriod = farmRecheckPeriod;
    m_subscribed.store(true, std::memory_order_relaxed);
    m_authorized.store(true, std::memory_order_relaxed);
    if (m_submit_hashrate)
        m_client_id = h256::random();
}

EthGetworkClient::~EthGetworkClient()
{
    p_client = nullptr;
}

void EthGetworkClient::connect()
{
    std::string uri = "http://" + m_conn->Host() + ":" + to_string(m_conn->Port());
    if (m_conn->Path().length())
    {
        uri += m_conn->Path();
    }
    p_client = new ::JsonrpcGetwork(new jsonrpc::HttpClient(uri));

    // Set we're connected
    m_connected.store(true, std::memory_order_relaxed);

    // Since we do not have a real connected state with getwork, we just fake it.
    if (m_onConnected)
    {
        m_onConnected();
    }

    // No need to worry about starting again.
    // Worker class prevents that
    startWorking();
}

void EthGetworkClient::disconnect()
{
    m_connected.store(false, std::memory_order_relaxed);

    // Since we do not have a real connected state with getwork, we just fake it.
    if (m_onDisconnected)
    {
        m_onDisconnected();
    }
}

void EthGetworkClient::submitHashrate(string const& rate)
{
    // Store the rate in temp var. Will be handled in workLoop
    // Hashrate submission does not need to be as quick as possible
    m_currentHashrateToSubmit = rate;
}

void EthGetworkClient::submitSolution(const Solution& solution)
{
    // Immediately send found solution without wait for loop
    if (m_connected.load(std::memory_order_relaxed))
    {
        try
        {
            std::chrono::steady_clock::time_point submit_start = std::chrono::steady_clock::now();
            bool accepted = p_client->eth_submitWork("0x" + toHex(solution.nonce),
                "0x" + toString(solution.work.header), "0x" + toString(solution.mixHash));
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
        catch (jsonrpc::JsonRpcException const& _e)
        {
            cwarn << "Failed to submit solution.";
            cwarn << boost::diagnostic_information(_e);
        }
    }
}

// Handles all getwork communication.
void EthGetworkClient::workLoop()
{
    while (!shouldStop())
    {
        if (m_connected.load(std::memory_order_relaxed))
        {
            // Get Work
            try
            {
                Json::Value v = p_client->eth_getWork();
                WorkPackage newWorkPackage;
                newWorkPackage.header = h256(v[0].asString());
                newWorkPackage.epoch = ethash::find_epoch_number(
                    ethash::hash256_from_bytes(h256{v[1].asString()}.data()));

                // Check if header changes so the new workpackage is really new
                if (newWorkPackage.header != m_prevWorkPackage.header)
                {
                    m_prevWorkPackage.header = newWorkPackage.header;
                    m_prevWorkPackage.epoch = newWorkPackage.epoch;
                    m_prevWorkPackage.boundary = h256(fromHex(v[2].asString()), h256::AlignRight);

                    if (m_onWorkReceived)
                    {
                        m_onWorkReceived(m_prevWorkPackage);
                    }
                }
            }
            catch (const jsonrpc::JsonRpcException&)
            {
                cwarn << "Failed getting work!";
                disconnect();
            }

            // Submit current hashrate if needed
            if (m_submit_hashrate && !m_currentHashrateToSubmit.empty())
            {
                try
                {
                    p_client->eth_submitHashrate(
                        m_currentHashrateToSubmit, "0x" + m_client_id.hex());
                }
                catch (const jsonrpc::JsonRpcException& _e)
                {
                    cwarn << "Failed to submit hashrate.";
                    cwarn << boost::diagnostic_information(_e);
                }
                m_currentHashrateToSubmit.clear();
            }
        }

        // Sleep for --farm-recheck defined period
        this_thread::sleep_for(chrono::milliseconds(m_farmRecheckPeriod));
    }

    // Since we do not have a real connected state with getwork, we just fake it.
    if (m_onDisconnected)
    {
        m_onDisconnected();
    }
}
