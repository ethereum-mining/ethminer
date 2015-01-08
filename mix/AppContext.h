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
	namespace solidity
	{
		class CompilerStack;
	}
}

namespace dev
{
namespace mix
{

class CodeModel;
/**
 * @brief Provides access to application scope variable.
 */

class AppContext: public QObject
{
	Q_OBJECT

public:
	AppContext(QQmlApplicationEngine* _engine);
	virtual ~AppContext();
	/// Get the current QQMLApplicationEngine instance.
	QQmlApplicationEngine* appEngine();
	/// Get code model
	CodeModel* codeModel() { return m_codeModel.get(); }
	/// Display an alert message.
	void displayMessageDialog(QString _title, QString _message);
	/// Load project settings
	void loadProject();
signals:
	void projectLoaded(QString const& _json);

private:
	QQmlApplicationEngine* m_applicationEngine; //owned by app
	std::unique_ptr<dev::WebThreeDirect> m_webThree;
	std::unique_ptr<CodeModel> m_codeModel;

public slots:
	/// Delete the current instance when application quit.
	void quitApplication() {}
	/// Write json to a settings file
	void saveProject(QString const& _json);
};

}
}
