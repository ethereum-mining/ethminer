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
/** @file AppContext.h
 * @author Yann yann@ethdev.com
 * @date 2014
 * Provides access to the current QQmlApplicationEngine which is used to add QML file on the fly.
 * In the future this class can be extended to add more variable related to the context of the application.
 * For now AppContext provides reference to:
 * - QQmlApplicationEngine
 * - dev::WebThreeDirect (and dev::eth::Client)
 * - KeyEventManager
 */

#pragma once

#include <memory>
#include <QQmlApplicationEngine>
#include <libsolidity/CompilerStack.h>
#include <libwebthree/WebThree.h>
#include "KeyEventManager.h"

namespace dev
{
	class WebThreeDirect;
	namespace solidity
	{
		class CompilerStack;
	}
}

namespace dev
{
namespace mix
{

/**
 * @brief Provides access to application scope variable.
 */
class AppContext: public QObject
{
	Q_OBJECT

public:
	AppContext(QQmlApplicationEngine* _engine);
	/// Get the current QQmlApplicationEngine instance.
	static AppContext* getInstance() { return Instance; }
	/// Renew QQMLApplicationEngine with a new instance.
	static void setApplicationContext(QQmlApplicationEngine* _engine);
	/// Get the current QQMLApplicationEngine instance.
	QQmlApplicationEngine* appEngine();
	/// Initialize KeyEventManager (used to handle key pressed event).
	void initKeyEventManager(QObject* _obj);
	/// Get the current KeyEventManager instance.
	KeyEventManager* getKeyEventManager();
	/// Get the current Compiler instance (used to parse and compile contract code).
	dev::solidity::CompilerStack* compiler();
	/// Display an alert message.
	void displayMessageDialog(QString _title, QString _message);

private:
	static AppContext* Instance;
	std::unique_ptr<QQmlApplicationEngine> m_applicationEngine;
	std::unique_ptr<dev::WebThreeDirect> m_webThree;
	std::unique_ptr<KeyEventManager> m_keyEventManager;
	std::unique_ptr<solidity::CompilerStack> m_compiler;

public slots:
	/// Delete the current instance when application quit.
	void quitApplication() { delete Instance; }
	/// Initialize components after the loading of the main QML view.
	void resourceLoaded(QObject* _obj, QUrl _url) { Q_UNUSED(_url); initKeyEventManager(_obj); }
};

}
}
