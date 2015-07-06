#include "ExecStats.h"

#include <iostream>
#include <iomanip>
#include <cassert>

#include "Utils.h"

namespace dev
{
namespace evmjit
{

void ExecStats::stateChanged(ExecState _state)
{
	if (!CHECK(m_state != ExecState::Finished))
		return;

	auto now = clock::now();
	if (_state != ExecState::Started)
	{
		assert(time[(int)m_state] == ExecStats::duration::zero());
		time[(int)m_state] = now - m_tp;
	}
	m_state = _state;
	m_tp = now;
}

namespace
{
struct StatsAgg
{
	using unit = std::chrono::microseconds;
	ExecStats::duration tot = ExecStats::duration::zero();
	ExecStats::duration min = ExecStats::duration::max();
	ExecStats::duration max = ExecStats::duration::zero();
	size_t count = 0;

	void update(ExecStats::duration _d)
	{
		++count;
		tot += _d;
		min = _d < min ? _d : min;
		max = _d > max ? _d : max;
	}

	void output(char const* _name, std::ostream& _os)
	{
		auto avg = tot / count;
		_os << std::setfill(' ')
			<< std::setw(12) << std::left  << _name
			<< std::setw(10) << std::right << std::chrono::duration_cast<unit>(tot).count()
			<< std::setw(10) << std::right << std::chrono::duration_cast<unit>(avg).count()
			<< std::setw(10) << std::right << std::chrono::duration_cast<unit>(min).count()
			<< std::setw(10) << std::right << std::chrono::duration_cast<unit>(max).count()
			<< std::endl;
	}
};

char const* getExecStateName(ExecState _state)
{
	switch (_state)
	{
	case ExecState::Started: return "Start";
	case ExecState::CacheLoad: return "CacheLoad";
	case ExecState::CacheWrite: return "CacheWrite";
	case ExecState::Compilation: return "Compilation";
	case ExecState::Optimization: return "Optimization";
	case ExecState::CodeGen: return "CodeGen";
	case ExecState::Execution: return "Execution";
	case ExecState::Return: return "Return";
	case ExecState::Finished: return "Finish";
	}
	return nullptr;
}
}

StatsCollector::~StatsCollector()
{
	if (stats.empty())
		return;

	std::cout << "        [us]     total       avg       min       max\n";
	for (int i = 0; i < (int)ExecState::Finished; ++i)
	{
		StatsAgg agg;
		for (auto&& s : stats)
			agg.update(s->time[i]);

		agg.output(getExecStateName(ExecState(i)), std::cout);
	}
}

}
}
