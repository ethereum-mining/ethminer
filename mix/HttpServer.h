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
/** @file HttpServer.h
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */

#pragma once

#include <QObject>
#include <QTcpServer>
#include <QUrl>
#include <QQmlParserStatus>

namespace dev
{
namespace mix
{

/// Simple http server for serving jsonrpc requests
class HttpRequest: public QObject
{
	Q_OBJECT
	/// Request url
	Q_PROPERTY(QUrl url MEMBER m_url CONSTANT)
	/// Request body contents
	Q_PROPERTY(QString content MEMBER m_content CONSTANT)

private:
	HttpRequest(QObject* _parent, QUrl const& _url, QString const& _content):
		QObject(_parent), m_url(_url), m_content(_content)
	{
	}

public:
	/// Set response for a request
	/// @param _response Response body. If no response is set, server returns status 200 with empty body
	Q_INVOKABLE void setResponse(QString const& _response) { m_response = _response; }
	/// Set response content type
	/// @param _contentType Response content type string. text/plain by default
	Q_INVOKABLE void setResponseContentType(QString const& _contentType) { m_responseContentType = _contentType ; }

private:
	QUrl m_url;
	QString m_content;
	QString m_response;
	QString m_responseContentType;
	friend class HttpServer;
};

class HttpServer: public QTcpServer, public QQmlParserStatus
{
	Q_OBJECT
	Q_DISABLE_COPY(HttpServer)
	Q_INTERFACES(QQmlParserStatus)

	/// Server url
	Q_PROPERTY(QUrl url READ url NOTIFY urlChanged)
	/// Server port
	Q_PROPERTY(int port READ port WRITE setPort NOTIFY portChanged)
	/// Listen for connections
	Q_PROPERTY(bool listen READ listen WRITE setListen NOTIFY listenChanged)
	/// Accept new connections
	Q_PROPERTY(bool accept READ accept WRITE setAccept NOTIFY acceptChanged)
	/// Error string if any
	Q_PROPERTY(QString errorString READ errorString NOTIFY errorStringChanged)

public:
	explicit HttpServer();
	virtual ~HttpServer();

	QUrl url() const;
	int port() const { return m_port; }
	void setPort(int _port);
	bool listen() const { return m_listen; }
	void setListen(bool _listen);
	bool accept() const { return m_accept; }
	void setAccept(bool _accept);
	QString errorString() const;

protected:
	virtual void classBegin() override;
	virtual void componentComplete() override;
	virtual void incomingConnection(qintptr _socket) override;

signals:
	void clientConnected(HttpRequest* _request);
	void errorStringChanged(QString const& _errorString);
	void urlChanged(QUrl const& _url);
	void portChanged(int _port);
	void listenChanged(bool _listen);
	void acceptChanged(bool _accept);

private:
	void init();
	void updateListening();
	void newConnection();
	void serverError();

private slots:
	void readClient();
	void discardClient();

private:
	int m_port;
	bool m_listen;
	bool m_accept;
	bool m_componentCompleted;
};

}
}
