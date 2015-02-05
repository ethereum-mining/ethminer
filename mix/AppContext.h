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
namespace mix
{

class CodeModel;
class ClientModel;
class FileIo;
/**
 * @brief Provides access to application scope variable.
 */

class AppContext: public QObject
{
	Q_OBJECT
	Q_PROPERTY(QString clipboard READ clipboard WRITE toClipboard NOTIFY clipboardChanged)

public:
	AppContext(QQmlApplicationEngine* _engine);
	virtual ~AppContext();
	/// Load the UI from qml files
	void load();
	/// Get the current QQMLApplicationEngine instance.
	QQmlApplicationEngine* appEngine();
	/// Get code model
	CodeModel* codeModel() { return m_codeModel.get(); }
	/// Get client model
	ClientModel* clientModel() { return m_clientModel.get(); }
	/// Display an alert message.
	void displayMessageDialog(QString _title, QString _message);
	/// Copy text to clipboard
	Q_INVOKABLE void toClipboard(QString _text);
	/// Get text from clipboard
	QString clipboard() const;

signals:
	/// Triggered once components have been loaded
	void appLoaded();
	void clipboardChanged();

private:
	QQmlApplicationEngine* m_applicationEngine; //owned by app
	std::unique_ptr<CodeModel> m_codeModel;
	std::unique_ptr<ClientModel> m_clientModel;
	std::unique_ptr<FileIo> m_fileIo;

public slots:
	/// Delete the current instance when application quit.
	void quitApplication() {}
};

}
}
