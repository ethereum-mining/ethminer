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
/** @file MiningView.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "MiningView.h"

#include <QtWidgets>
#include <QtCore>
#include <libethereum/Client.h>
#include "Grapher.h"

using namespace std;
using namespace lb;

// do *not* use eth since unsigned conflicts with Qt's global unit definition
// using namespace dev;
using namespace dev::eth;

// types

using dev::eth::MineInfo;
using dev::eth::WorkingProgress;

// functions
using dev::toString;
using dev::trimFront;

string id(float _y) { return toString(_y); }
string s(float _x){ return toString(round(_x * 1000) / 1000) + (!_x ? "s" : ""); }
string sL(float _x, float _y) { return toString(round(_x * 1000)) + "s (" + toString(_y) + ")"; }

MiningView::MiningView(QWidget* _p): QWidget(_p)
{
}

void MiningView::appendStats(list<MineInfo> const& _i, WorkingProgress const& _p)
{
	(void)_p;
	if (_i.empty())
		return;

/*	unsigned o = m_values.size();
	for (MineInfo const& i: _i)
	{
		m_values.push_back(i.best);
		m_lastBest = min(m_lastBest, i.best);
		m_bests.push_back(m_lastBest);
		m_reqs.push_back(i.requirement);
		if (i.completed)
		{
			m_completes.push_back(o);
			m_resets.push_back(o);
			m_haveReset = false;
			m_lastBest = 1e99;
		}
		++o;
	}
	if (m_haveReset)
	{
		m_resets.push_back(o - 1);
		m_lastBest = 1e99;
		m_haveReset = false;
	}

	o = max<int>(0, (int)m_values.size() - (int)m_duration);
	trimFront(m_values, o);
	trimFront(m_bests, o);
	trimFront(m_reqs, o);

	for (auto& i: m_resets)
		i -= o;
	m_resets.erase(remove_if(m_resets.begin(), m_resets.end(), [](int i){return i < 0;}), m_resets.end());
	for (auto& i: m_completes)
		i -= o;
	m_completes.erase(remove_if(m_completes.begin(), m_completes.end(), [](int i){return i < 0;}), m_completes.end());

	m_progress = _p;
	update();*/
}

void MiningView::resetStats()
{
	m_haveReset = true;
}

void MiningView::paintEvent(QPaintEvent*)
{
	/*
	Grapher g;
	QPainter p(this);

	g.init(&p, make_pair(0.f, max((float)m_duration * 0.1f, (float)m_values.size() * 0.1f)), make_pair(0.0f, 255.f - ((float)m_progress.requirement - 4.0f)), s, id, sL);
	g.drawAxes();
	g.setDataTransform(0.1f, 0, -1.0f, 255.f);

	g.drawLineGraph(m_values, QColor(192, 192, 192));
	g.drawLineGraph(m_bests, QColor(128, 128, 128));
	g.drawLineGraph(m_reqs, QColor(128, 64, 64));
	for (auto r: m_resets)
		g.ruleY(r - 1, QColor(128, 128, 128));
	for (auto r: m_completes)
		g.ruleY(r, QColor(192, 64, 64));
	*/
}
