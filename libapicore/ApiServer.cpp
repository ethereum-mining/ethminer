#include "ApiServer.h"
#include "BuildInfo.h"

ApiServer::ApiServer(AbstractServerConnector *conn, serverVersion_t type, Farm &farm, bool &readonly) : AbstractServer(*conn, type), m_farm(farm)
{
	this->bindAndAddMethod(Procedure("miner_getstat1", PARAMS_BY_NAME, JSON_OBJECT, NULL), &ApiServer::getMinerStat1);
	this->bindAndAddMethod(Procedure("miner_getstathr", PARAMS_BY_NAME, JSON_OBJECT, NULL), &ApiServer::getMinerStatHR);	
	if (!readonly) {
		this->bindAndAddMethod(Procedure("miner_restart", PARAMS_BY_NAME, JSON_OBJECT, NULL), &ApiServer::doMinerRestart);
		this->bindAndAddMethod(Procedure("miner_reboot", PARAMS_BY_NAME, JSON_OBJECT, NULL), &ApiServer::doMinerReboot);
	}
}

void ApiServer::getMinerStat1(const Json::Value& request, Json::Value& response)
{
	(void) request; // unused
	
	auto runningTime = std::chrono::duration_cast<std::chrono::minutes>(steady_clock::now() - this->m_farm.farmLaunched());
	
	SolutionStats s = m_farm.getSolutionStats();
	WorkingProgress p = m_farm.miningProgress(true);
	
	ostringstream totalMhEth; 
	ostringstream totalMhDcr; 
	ostringstream detailedMhEth;
	ostringstream detailedMhDcr;
	ostringstream tempAndFans;
	ostringstream poolAddresses;
	ostringstream invalidStats;
	
	totalMhEth << std::fixed << std::setprecision(0) << (p.rate() / 1000.0f) << ";" << s.getAccepts() << ";" << s.getRejects();
	totalMhDcr << "0;0;0"; // DualMining not supported
	invalidStats << s.getFailures() << ";0"; // Invalid + Pool switches
    poolAddresses << m_farm.get_pool_addresses(); 
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
	response[7] = poolAddresses.str();           // current mining pool. For dual mode, there will be two pools here.
	response[8] = invalidStats.str();            // number of ETH invalid shares, number of ETH pool switches, number of DCR invalid shares, number of DCR pool switches.
}

void ApiServer::getMinerStatHR(const Json::Value& request, Json::Value& response)
{
	(void) request; // unused
	
	//TODO:give key-value format
	auto runningTime = std::chrono::duration_cast<std::chrono::minutes>(steady_clock::now() - this->m_farm.farmLaunched());
	
	SolutionStats s = m_farm.getSolutionStats();
	WorkingProgress p = m_farm.miningProgress(true,true);
	
	ostringstream version; 
	ostringstream runtime; 
	ostringstream ethhashrate; 
	ostringstream ethshares; 
	ostringstream ethrejects;  
	ostringstream ethinvalid; 
	ostringstream ethpoolsw;
	ostringstream dcrhashrate; 
	ostringstream dcrshares; 
	ostringstream dcrrejects;   
	ostringstream dcrinvalid; 
	ostringstream dcrpoolsw;
	ostringstream totalMhEth; 
	ostringstream totalMhDcr; 
	Json::Value detailedMhEth;
	Json::Value detailedMhDcr;
	ostringstream detailedMhEthStr;
	ostringstream detailedMhDcrStr;
	Json::Value temps;
	Json::Value fans;
	Json::Value powers;
	ostringstream tempStr;
	ostringstream fanStr;
	ostringstream powerStr;
	ostringstream poolAddresses;
	ostringstream invalidStats;
	
	version << ETH_PROJECT_VERSION ;
	runtime << toString(runningTime.count());

	ethhashrate << std::fixed << std::setprecision(1) << (p.rate() / 1000.0f) << "h/s";
	ethshares << std::fixed << std::setprecision(1) << s.getAccepts();
	ethrejects << std::fixed << std::setprecision(1) << s.getRejects();	
	dcrhashrate << std::fixed << std::setprecision(1) << 0.0 << "h/s"; // Not supported 
	dcrshares << std::fixed << std::setprecision(1) << 0.0; // Not supported 
	dcrrejects << std::fixed << std::setprecision(1) << 0.0; // Not supported 
	ethinvalid << s.getFailures() ; // Invalid 
	ethpoolsw << 0; // Pool switches
	dcrinvalid << 0; // Not supported 
	dcrpoolsw << 0; // Not supported 
    poolAddresses << m_farm.get_pool_addresses(); 
	
	int gpuIndex = 0;
	int numGpus = p.minersHashes.size();
	for (auto const& i: p.minersHashes)
	{
		detailedMhEthStr.str("");
		detailedMhDcrStr.str("");
		detailedMhEthStr << std::fixed << std::setprecision(1) << (p.minerRate(i) / 1000.0f) << "h/s";
		detailedMhDcrStr << "off"; // DualMining not supported
		detailedMhEth[gpuIndex] = detailedMhEthStr.str();
		detailedMhDcr[gpuIndex] = detailedMhDcrStr.str();
		gpuIndex++;
	}

	gpuIndex = 0;
	numGpus = p.minerMonitors.size();
	for (auto const& i : p.minerMonitors)
	{
		tempStr.str("");
		fanStr.str("");
		powerStr.str("");
		tempStr << std::fixed << std::setprecision(1) << i.tempC << "C"; // Fetching Temps 
		fanStr << std::fixed << std::setprecision(1) << i.fanP << "%"; // Fetching Fans
		powerStr << std::fixed << std::setprecision(1) << i.powerW << "W"; // Fetching Power
		temps[gpuIndex] = tempStr.str(); // Fetching Temps 
		fans[gpuIndex] = fanStr.str(); // Fetching Fans
		powers[gpuIndex] = powerStr.str(); // Fetching Power
		gpuIndex++;
	}

	response["version"] = version.str();		// miner version.
	response["runtime"] = runtime.str();		// running time, in minutes.
	// total ETH hashrate in MH/s, number of ETH shares, number of ETH rejected shares.
	response["ethhashrate"] = ethhashrate.str();
	response["ethhashrates"] = detailedMhEth;  
	response["ethshares"] 	= ethshares.str(); 
	response["ethrejected"] = ethrejects.str();   
	response["ethinvalid"] 	= ethinvalid.str(); 
	response["ethpoolsw"] 	= ethpoolsw.str();          
	// DCR not supported
	// response["dcrhashrate"] = dcrhashrate.str();
	// response["dcrhashrates"] = detailedMhDcr;   
	// response["dcrshares"] 	= dcrshares.str(); 
	// response["dcrrejected"] = dcrrejects.str();       
	// response["dcrinvalid"] 	= dcrinvalid.str(); 
	// response["dcrpoolsw"] 	= dcrpoolsw.str();       
	// Hardware Info
	response["temperatures"] = temps;             		// Temperatures(C) for all GPUs
	response["fanpercentages"] = fans;             		// Fans speed(%) for all GPUs
	response["powerusages"] = powers;         			// Power Usages(W) for all GPUs
	response["pooladdrs"] = poolAddresses.str();        // current mining pool. For dual mode, there will be two pools here.
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
