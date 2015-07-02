#pragma once

#include <memory>
#include <vector>
#include <string>
#include <chrono>

namespace dev
{
namespace evmjit
{

enum class ExecState
{
	Started,
	CacheLoad,
	CacheWrite,
	Compilation,
	Optimization,
	CodeGen,
	Execution,
	Return,
	Finished
};

class JITListener
{
public:
	JITListener() = default;
	JITListener(JITListener const&) = delete;
	JITListener& operator=(JITListener) = delete;
	virtual ~JITListener() {}

	virtual void executionStarted() {}
	virtual void executionEnded() {}

	virtual void stateChanged(ExecState) {}
};

class ExecStats : public JITListener
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
