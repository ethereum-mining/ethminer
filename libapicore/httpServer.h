#pragma once

#include <thread>
#include <libethcore/Farm.h>

class httpServer
{
public:
    void run(unsigned short port, dev::eth::Farm* farm, bool show_hwmonitors, bool show_power);
    void run_thread();
    void getstat1(stringstream& ss);

    dev::eth::Farm* m_farm;
    std::string m_port;
private:
    void tableHeader(stringstream& ss, unsigned columns);
	bool m_show_hwmonitors;
    bool m_show_power;
};

extern httpServer http_server;

