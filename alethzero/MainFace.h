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

#define DEV_AZ_NOTE_PLUGIN(ClassName) \
	static bool s_notePlugin = [](){ MainFace::notePlugin([](MainFace* m){ return new ClassName(m); }); return true; }()

class Plugin;
class MainFace;
class Main;

using WatchHandler = std::function<void(dev::eth::LocalisedLogEntries const&)>;

class AccountNamer
{
	friend class Main;

public:
	virtual std::string toName(Address const&) const { return std::string(); }
	virtual Address toAddress(std::string const&) const { return Address(); }
	virtual Addresses knownAddresses() const { return Addresses(); }

protected:
	void noteKnownChanged();
	void noteNamesChanged();

private:
	MainFace* m_main = nullptr;
};

class MainFace: public QMainWindow, public Context
{
	Q_OBJECT

public:
	explicit MainFace(QWidget* _parent = nullptr): QMainWindow(_parent) {}

	static void notePlugin(std::function<Plugin*(MainFace*)> const& _new);

	void adoptPlugin(Plugin* _p);
	void killPlugins();

	void allChange();

	using Context::render;

	// TODO: tidy - all should be references that throw if module unavailable.
	// TODO: provide a set of available web3 modules.
	virtual dev::WebThreeDirect* web3() const = 0;
	virtual dev::eth::Client* ethereum() const = 0;
	virtual std::shared_ptr<dev::shh::WhisperHost> whisper() const = 0;

	virtual unsigned installWatch(dev::eth::LogFilter const& _tf, WatchHandler const& _f) = 0;
	virtual unsigned installWatch(dev::h256 const& _tf, WatchHandler const& _f) = 0;
	virtual void uninstallWatch(unsigned _id) = 0;

	// Account naming API
	virtual void install(AccountNamer* _adopt) = 0;
	virtual void uninstall(AccountNamer* _kill) = 0;
	virtual void noteKnownAddressesChanged(AccountNamer*) = 0;
	virtual void noteAddressNamesChanged(AccountNamer*) = 0;
	virtual Address toAddress(std::string const&) const = 0;
	virtual std::string toName(Address const&) const = 0;
	virtual Addresses allKnownAddresses() const = 0;

	virtual void noteSettingsChanged() = 0;

protected:
	template <class F> void forEach(F const& _f) { for (auto const& p: m_plugins) _f(p.second); }
	std::shared_ptr<Plugin> takePlugin(std::string const& _name) { auto it = m_plugins.find(_name); std::shared_ptr<Plugin> ret; if (it != m_plugins.end()) { ret = it->second; m_plugins.erase(it); } return ret; }

	static std::vector<std::function<Plugin*(MainFace*)>>* s_linkedPlugins;

signals:
	void knownAddressesChanged();
	void addressNamesChanged();
	void keyManagerChanged();

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
	MainFace* main() const { return m_main; }
	QDockWidget* dock(Qt::DockWidgetArea _area = Qt::RightDockWidgetArea, QString _title = QString());
	void addToDock(Qt::DockWidgetArea _area, QDockWidget* _dockwidget, Qt::Orientation _orientation);
	void addAction(QAction* _a);
	QAction* addMenuItem(QString _name, QString _menuName, bool _separate = false);

	virtual void onAllChange() {}
	virtual void readSettings(QSettings const&) {}
	virtual void writeSettings(QSettings&) {}

private:
	MainFace* m_main = nullptr;
	std::string m_name;
	QDockWidget* m_dock = nullptr;
};

class AccountNamerPlugin: public Plugin, public AccountNamer
{
protected:
	AccountNamerPlugin(MainFace* _m, std::string const& _name): Plugin(_m, _name) { main()->install(this); }
	~AccountNamerPlugin() { main()->uninstall(this); }
};

}

}
