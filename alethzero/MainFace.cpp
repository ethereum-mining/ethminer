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
/** @file MainFace.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "MainFace.h"
using namespace std;
using namespace dev;
using namespace az;

Plugin::Plugin(MainFace* _f, std::string const& _name):
	m_main(_f),
	m_name(_name)
{
	_f->adoptPlugin(this);
}

QDockWidget* Plugin::dock(Qt::DockWidgetArea _area, QString _title)
{
	if (_title.isEmpty())
		_title = QString::fromStdString(m_name);
	if (!m_dock)
	{
		m_dock = new QDockWidget(_title, m_main);
		m_main->addDockWidget(_area, m_dock);
		m_dock->setFeatures(QDockWidget::AllDockWidgetFeatures | QDockWidget::DockWidgetVerticalTitleBar);
	}
	return m_dock;
}

void Plugin::addToDock(Qt::DockWidgetArea _area, QDockWidget* _dockwidget, Qt::Orientation _orientation)
{
	m_main->addDockWidget(_area, _dockwidget, _orientation);
}

void Plugin::addAction(QAction* _a)
{
	m_main->addAction(_a);
}

void MainFace::adoptPlugin(Plugin* _p)
{
	m_plugins[_p->name()] = shared_ptr<Plugin>(_p);
}

void MainFace::killPlugins()
{
	m_plugins.clear();
}

void MainFace::allChange()
{
	for (auto const& p: m_plugins)
		p.second->onAllChange();
}
