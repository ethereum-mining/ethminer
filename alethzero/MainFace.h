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
#include <functional>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QAction>
#include <QtWidgets/QDockWidget>
#include <libevm/ExtVMFace.h>
#include "Context.h"

class QSettings;

namespace dev
{

class WebThreeDirect;
namespace eth { class Client; class LogFilter; }
namespace shh { class WhisperHost; }

namespace az
{

class Plugin;

using WatchHandler = std::function<void(dev::eth::LocalisedLogEntries const&)>;

class MainFace: public QMainWindow, public Context
{
public:
	explicit MainFace(QWidget* _parent = nullptr): QMainWindow(_parent) {}

	void adoptPlugin(Plugin* _p);
	void killPlugins();

	void allChange();

	// TODO: tidy - all should be references that throw if module unavailable.
	// TODO: provide a set of available web3 modules.
	virtual dev::WebThreeDirect* web3() const = 0;
	virtual dev::eth::Client* ethereum() const = 0;
	virtual std::shared_ptr<dev::shh::WhisperHost> whisper() const = 0;

	virtual unsigned installWatch(dev::eth::LogFilter const& _tf, WatchHandler const& _f) = 0;
	virtual unsigned installWatch(dev::h256 const& _tf, WatchHandler const& _f) = 0;

protected:
	template <class F> void forEach(F const& _f) { for (auto const& p: m_plugins) _f(p.second); }
	std::shared_ptr<Plugin> takePlugin(std::string const& _name) { auto it = m_plugins.find(_name); std::shared_ptr<Plugin> ret; if (it != m_plugins.end()) { ret = it->second; m_plugins.erase(it); } return ret; }

private:
	std::unordered_map<std::string, std::shared_ptr<Plugin>> m_plugins;
};

class Plugin
{
public:
	Plugin(MainFace* _f, std::string const& _name);
	virtual ~Plugin() {}

	std::string const& name() const { return m_name; }

	dev::WebThreeDirect* web3() const { return m_main->web3(); }
	dev::eth::Client* ethereum() const { return m_main->ethereum(); }
	std::shared_ptr<dev::shh::WhisperHost> whisper() const { return m_main->whisper(); }
	MainFace* main() { return m_main; }
	QDockWidget* dock(Qt::DockWidgetArea _area = Qt::RightDockWidgetArea, QString _title = QString());
	void addToDock(Qt::DockWidgetArea _area, QDockWidget* _dockwidget, Qt::Orientation _orientation);
	void addAction(QAction* _a);

	virtual void onAllChange() {}
	virtual void readSettings(QSettings const&) {}
	virtual void writeSettings(QSettings&) {}

private:
	MainFace* m_main = nullptr;
	std::string m_name;
	QDockWidget* m_dock = nullptr;
};

}

}
