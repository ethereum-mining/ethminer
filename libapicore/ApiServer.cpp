#include "ApiServer.h"
#include "BuildInfo.h"

ApiServer::ApiServer(AbstractServerConnector *conn, serverVersion_t type, Farm &farm, bool &readonly) : AbstractServer(*conn, type), m_farm(farm)
{
	this->bindAndAddMethod(Procedure("miner_getstat1", PARAMS_BY_NAME, JSON_OBJECT, NULL), &ApiServer::getMinerStat1);
	if (!readonly) {
		this->bindAndAddMethod(Procedure("miner_restart", PARAMS_BY_NAME, JSON_OBJECT, NULL), &ApiServer::doMinerRestart);
		this->bindAndAddMethod(Procedure("miner_reboot", PARAMS_BY_NAME, JSON_OBJECT, NULL), &ApiServer::doMinerReboot);
	}
}

void ApiServer::getMinerStat1(const Json::Value& request, Json::Value& response)
{
	(void) request; // unused
	
	auto runningTime = std::chrono::duration_cast<std::chrono::minutes>(steady_clock::now() - this->m_farm.farmLaunched());
	
	SolutionStats s = this->m_farm.getSolutionStats();
	WorkingProgress p = this->m_farm.miningProgress(true);
	
	ostringstream totalMhEth; 
	ostringstream totalMhDcr; 
	ostringstream detailedMhEth;
	ostringstream detailedMhDcr;
	ostringstream tempAndFans;
	ostringstream invalidStats;
	
	totalMhEth << std::fixed << std::setprecision(0) << (p.rate() / 1000.0f) << ";" << s.getAccepts() << ";" << s.getRejects();
	totalMhDcr << "0;0;0"; // DualMining not supported
	invalidStats << s.getFailures() << ";0"; // Invalid + Pool switches
	invalidStats << ";0;0"; // DualMining not supported
	
	int gpuIndex = 0;
	int numGpus = p.minersHashes.size();
	for (auto const& i: p.minersHashes)
	{
		detailedMhEth << std::fixed << std::setprecision(0) << (p.minerRate(i) / 1000.0f) << (((numGpus -1) > gpuIndex) ? ";" : "");
		detailedMhDcr << "off" << (((numGpus -1) > gpuIndex) ? ";" : ""); // DualMining not supported
		gpuIndex++;
	}

	gpuIndex = 0;
	numGpus = p.minerMonitors.size();
	for (auto const& i : p.minerMonitors)
	{
		tempAndFans << i.tempC << ";" << i.fanP << (((numGpus - 1) > gpuIndex) ? "; " : ""); // Fetching Temp and Fans
		gpuIndex++;
	}

	response[0] = ETH_PROJECT_VERSION;           //miner version.
	response[1] = toString(runningTime.count()); // running time, in minutes.
	response[2] = totalMhEth.str();              // total ETH hashrate in MH/s, number of ETH shares, number of ETH rejected shares.
	response[3] = detailedMhEth.str();           // detailed ETH hashrate for all GPUs.
	response[4] = totalMhDcr.str();              // total DCR hashrate in MH/s, number of DCR shares, number of DCR rejected shares.
	response[5] = detailedMhDcr.str();           // detailed DCR hashrate for all GPUs.
	response[6] = tempAndFans.str();             // Temperature and Fan speed(%) pairs for all GPUs.
	response[7] = "";                            // current mining pool. For dual mode, there will be two pools here.
	response[8] = invalidStats.str();            // number of ETH invalid shares, number of ETH pool switches, number of DCR invalid shares, number of DCR pool switches.
}

void ApiServer::doMinerRestart(const Json::Value& request, Json::Value& response)
{
	(void) request; // unused
	(void) response; // unused
	
	this->m_farm.restart();
}

void ApiServer::doMinerReboot(const Json::Value& request, Json::Value& response)
{
	(void) request; // unused
	(void) response; // unused
	
	// Not supported
}