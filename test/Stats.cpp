/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "Stats.h"

#include <iterator>
#include <numeric>
#include <fstream>

namespace dev
{
namespace test
{

Stats& Stats::get()
{
	static Stats instance;
	return instance;
}

void Stats::suiteStarted(std::string const& _name)
{
	m_currentSuite = _name;
}

void Stats::testStarted(std::string const& _name)
{
	m_currentTest = _name;
	m_tp = clock::now();
}

void Stats::testFinished()
{
	m_stats.push_back({clock::now() - m_tp, m_currentSuite + "/" + m_currentTest});
}

std::ostream& operator<<(std::ostream& out, Stats::clock::duration const& d)
{
	return out << std::setw(10) << std::right << std::chrono::duration_cast<std::chrono::microseconds>(d).count() << " us";
}

Stats::~Stats()
{
	if (m_stats.empty())
		return;

	std::sort(m_stats.begin(), m_stats.end(), [](Stats::Item const& a, Stats::Item const& b){
		return a.duration < b.duration;
	});

	auto& out = std::cout;
	auto itr = m_stats.begin();
	auto min = *itr;
	auto max = *m_stats.rbegin();
	std::advance(itr, m_stats.size() / 2);
	auto med = *itr;
	auto tot = std::accumulate(m_stats.begin(), m_stats.end(), clock::duration{}, [](clock::duration const& a, Stats::Item const& v)
	{
		return a + v.duration;
	});

	out << "\nSTATS:\n\n" << std::setfill(' ');

	if (Options::get().statsOutFile == "out")
	{
		for (auto&& s: m_stats)
			out << "  " << std::setw(40) << std::left << s.name.substr(0, 40) << s.duration << " \n";
		out << "\n";
	}
	else if (!Options::get().statsOutFile.empty())
	{
		// Output stats to file
		std::ofstream file{Options::get().statsOutFile};
		for (auto&& s: m_stats)
			file << s.name << "\t" << std::chrono::duration_cast<std::chrono::microseconds>(s.duration).count() << "\n";
	}

	out	<< "  tot: " << tot << "\n"
		<< "  avg: " << (tot / m_stats.size()) << "\n\n"
		<< "  min: " << min.duration << " (" << min.name << ")\n"
		<< "  med: " << med.duration << " (" << med.name << ")\n"
		<< "  max: " << max.duration << " (" << max.name << ")\n";
}

}
}
