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
/** @file QWebThree.cpp
 * @authors:
 *   Gav Wood <i@gavwood.com>
 *   Marek Kotewicz <marek@ethdev.com>
 * @date 2014
 */

#include <QtCore/QtCore>
#include "QWebThree.h"

using namespace std;

QWebThree::QWebThree(QObject* _p): QObject(_p)
{
	moveToThread(_p->thread());
}

QWebThree::~QWebThree()
{
	clientDieing();
}

void QWebThree::clientDieing()
{
	this->disconnect();
}

QString QWebThree::callMethod(QString _json)
{
	emit processData(_json, ""); // it's synchronous
	return m_response;
}

void QWebThree::onDataProcessed(QString _json, QString)
{
	QJsonObject f = QJsonDocument::fromJson(_json.toUtf8()).object();
	syncResponse(QString::fromUtf8(QJsonDocument(f).toJson()));
}

void QWebThree::syncResponse(QString _json)
{
	m_response = _json;
}

QWebThreeConnector::QWebThreeConnector()
{
}

QWebThreeConnector::~QWebThreeConnector()
{
	StopListening();
}

void QWebThreeConnector::setQWeb(QWebThree* _q)
{
	m_qweb = _q;
	if (m_isListening)
	{
		StopListening();
		StartListening();
	}
}

bool QWebThreeConnector::StartListening()
{
	m_isListening = true;
	if (m_qweb)
	{
		connect(m_qweb, SIGNAL(processData(QString, QString)), this, SLOT(onProcessData(QString, QString)));
		connect(this, SIGNAL(dataProcessed(QString, QString)), m_qweb, SLOT(onDataProcessed(QString, QString)));
	}
	return true;
}

bool QWebThreeConnector::StopListening()
{
	this->disconnect();
	return true;
}

bool QWebThreeConnector::SendResponse(std::string const& _response, void* _addInfo)
{
	emit dataProcessed(QString::fromStdString(_response), *(QString*)_addInfo);
	return true;
}

void QWebThreeConnector::onProcessData(QString const& _json, QString const& _addInfo)
{
	OnRequest(_json.toStdString(), (void*)&_addInfo);
}

