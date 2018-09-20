#pragma once

#include <iostream>

#include "jsonrpc_getwork.h"
#include <jsonrpccpp/client/connectors/httpclient.h>

#include <libdevcore/Worker.h>

#include "../PoolClient.h"

using namespace std;
using namespace dev;
using namespace eth;

class EthGetworkClient : public PoolClient, Worker
{
public:
    EthGetworkClient(unsigned farmRecheckPeriod, bool submitHashrate);
    ~EthGetworkClient();

    void connect() override;
    void disconnect() override;

    bool isConnected() override { return m_connected; }
    bool isPendingState() override { return false; }

    string ActiveEndPoint() override { return ""; };

    void submitHashrate(string const& rate) override;
    void submitSolution(const Solution& solution) override;

private:
    void workLoop() override;
    unsigned m_farmRecheckPeriod = 500;

    string m_currentHashrateToSubmit = "";

    h256 m_client_id;
    JsonrpcGetwork* p_client = nullptr;
    WorkPackage m_prevWorkPackage;

    // Hashrate submission is optional
    bool m_submit_hashrate;
};
