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
/** @file MainFace.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <memory>
#include <map>
#include <string>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QAction>
#include <QtWidgets/QDockWidget>
#include "Context.h"

namespace dev
{

namespace web3 { class WebThreeDirect; }
namespace eth { class Client; }
namespace shh { class WhisperHost; }

namespace az
{

class Plugin;

class MainFace: public QMainWindow, public Context
{
public:
	explicit MainFace(QWidget* _parent = nullptr): QMainWindow(_parent) {}

	void adoptPlugin(Plugin* _p) { m_plugins.insert(_p->name(), std::shared_ptr<Plugin>(_p)); }
	void killPlugins();

	void allChange();

	// TODO: tidy - all should be references that throw if module unavailable.
	// TODO: provide a set of available web3 modules.
	virtual dev::web3::WebThreeDirect* web3() const = 0;
	virtual dev::eth::Client* ethereum() const = 0;
	virtual std::shared_ptr<dev::shh::WhisperHost> whisper() const = 0;

private:
	std::unordered_map<std::string, std::shared_ptr<Plugin>> m_plugins;
};

class Plugin
{
public:
	Plugin(MainFace* _f, std::string const& _name);
	virtual ~Plugin() {}

	std::string const& name() const { return m_name; }

	dev::web3::WebThreeDirect* web3() const { return m_main->web3(); }
	dev::eth::Client* ethereum() const { return m_main->ethereum(); }
	std::shared_ptr<dev::shh::WhisperHost> whisper() const { return m_main->whisper(); }
	MainFace* main() { return m_main; }
	QDockWidget* dock(Qt::DockWidgetArea _area = Qt::RightDockWidgetArea, QString _title = QString());
	void addToDock(Qt::DockWidgetArea _area, QDockWidget* _dockwidget, Qt::Orientation _orientation);
	void addAction(QAction* _a);

	virtual void onAllChange() {}

private:
	MainFace* m_main;
	std::string m_name;
	QDockWidget* m_dock;
};

}

}
