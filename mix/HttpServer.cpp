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
/** @file HttpServer.cpp
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */

#include <memory>
#include <QTcpSocket>
#include "HttpServer.h"


using namespace dev::mix;

HttpServer::HttpServer()
	: m_port(0) , m_listen(false) , m_accept(true) , m_componentCompleted(true)
{
}

HttpServer::~HttpServer()
{
	setListen(false);
}

void HttpServer::classBegin()
{
	m_componentCompleted = false;
}

void HttpServer::componentComplete()
{
	init();
	m_componentCompleted = true;
}

QUrl HttpServer::url() const
{
	QUrl url;
	url.setPort(m_port);
	url.setHost("localhost");
	url.setScheme("http");
	return url;
}

void HttpServer::setPort(int _port)
{
	if (_port == m_port)
		return;

	m_port = _port;
	emit portChanged(_port);
	emit urlChanged(url());

	if (m_componentCompleted && this->isListening())
		updateListening();
}

QString HttpServer::errorString() const
{
	return QTcpServer::errorString();
}

void HttpServer::setListen(bool _listen)
{
	if (_listen == m_listen)
		return;

	m_listen = _listen;
	emit listenChanged(_listen);

	if (m_componentCompleted)
		updateListening();
}

void HttpServer::setAccept(bool _accept)
{
	if (_accept == m_accept)
		return;

	m_accept = _accept;
	emit acceptChanged(_accept);
}

void HttpServer::init()
{
	updateListening();
}

void HttpServer::updateListening()
{
	if (this->isListening())
		this->close();

	if (!m_listen || QTcpServer::listen(QHostAddress::LocalHost, m_port))
		return;
}

void HttpServer::incomingConnection(qintptr _socket)
{
	if (!m_accept)
		return;

	QTcpSocket* s = new QTcpSocket(this);
	connect(s, SIGNAL(readyRead()), this, SLOT(readClient()));
	connect(s, SIGNAL(disconnected()), this, SLOT(discardClient()));
	s->setSocketDescriptor(_socket);
}

void HttpServer::readClient()
{
	if (!m_accept)
		return;

	QTcpSocket* socket = (QTcpSocket*)sender();
	try
	{
		if (socket->canReadLine())
		{
			QString hdr = QString(socket->readLine());
			if (hdr.startsWith("POST") || hdr.startsWith("GET"))
			{
				QUrl url(hdr.split(' ')[1]);
				QString l;
				do
					l = socket->readLine();
				while (!(l.isEmpty() || l == "\r" || l == "\r\n"));

				QString content = socket->readAll();
				std::unique_ptr<HttpRequest> request(new HttpRequest(this, url, content));
				clientConnected(request.get());
				QTextStream os(socket);
				os.setAutoDetectUnicode(true);
				QString q;
				///@todo: allow setting response content-type, charset, etc
				os << "HTTP/1.0 200 Ok\r\n";
				if (!request->m_responseContentType.isEmpty())
					os << "Content-Type: " << request->m_responseContentType << "; ";
				os << "charset=\"utf-8\"\r\n\r\n";
				os << request->m_response;
			}
		}
	}
	catch(...)
	{
		delete socket;
		throw;
	}
	socket->close();
	if (socket->state() == QTcpSocket::UnconnectedState)
		delete socket;
}

void HttpServer::discardClient()
{
	QTcpSocket* socket = (QTcpSocket*)sender();
	socket->deleteLater();
}
