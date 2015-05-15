#pragma once

#include <memory>
#include <vector>
#include <string>
#include <chrono>

#include "ExecutionEngine.h"

namespace dev
{
namespace evmjit
{

class ExecStats : public ExecutionEngineListener
{
public:
	using clock = std::chrono::high_resolution_clock;
	using duration = clock::duration;
	using time_point = clock::time_point;

	std::string id;
	duration time[(int)ExecState::Finished] = {};

	void stateChanged(ExecState _state) override;

private:
	ExecState m_state = {};
	time_point m_tp = {};

};


class StatsCollector
{
public:
	std::vector<std::unique_ptr<ExecStats>> stats;

	~StatsCollector();
};

}
}
