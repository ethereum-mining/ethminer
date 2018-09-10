#pragma once

#include <thread>

#include <libethcore/Farm.h>
#include <libpoolprotocols/PoolManager.h>

class httpServer
{
public:
    void run(string address, uint16_t port, dev::eth::Farm* farm, dev::eth::PoolManager* manager,
        bool show_hwmonitors, bool show_power);
    void run_thread();
    void getstat1(stringstream& ss);

    dev::eth::Farm* m_farm;
    dev::eth::PoolManager* m_manager;
    std::string m_port;

private:
    void tableHeader(stringstream& ss, unsigned columns);
    bool m_show_hwmonitors;
    bool m_show_power;
};

extern httpServer http_server;
