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
#include <QUrl>
#include <QObject>

class QQmlApplicationEngine;

namespace dev
{

class WebThreeDirect;

namespace eth
{
	class Client;
}

namespace mix
{

class CodeModel;
class KeyEventManager;
/**
 * @brief Provides access to application scope variable.
 */

class AppContext : public QObject
{
	Q_OBJECT

public:
	AppContext(QQmlApplicationEngine* _engine);
	~AppContext();
	QQmlApplicationEngine* appEngine();
	/// Initialize KeyEventManager (used to handle key pressed event).
	void initKeyEventManager(QObject* _obj);
	/// Get the current KeyEventManager instance.
	KeyEventManager* getKeyEventManager();
	/// Get code model
	CodeModel* codeModel() { return m_codeModel.get(); }
	/// Display an alert message.
	void displayMessageDialog(QString _title, QString _message);

private:
	QQmlApplicationEngine* m_applicationEngine; //owned by app
	std::unique_ptr<dev::WebThreeDirect> m_webThree;
	std::unique_ptr<KeyEventManager> m_keyEventManager;
	std::unique_ptr<CodeModel> m_codeModel;

public slots:
	/// Delete the current instance when application quit.
	void quitApplication() {}
	/// Initialize components after the loading of the main QML view.
	void resourceLoaded(QObject* _obj, QUrl _url);
};

}
}
