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
/** @file QEthereum.cpp
 * @authors:
 *   Gav Wood <i@gavwood.com>
 *   Marek Kotewicz <marek@ethdev.com>
 * @date 2014
 */

#include <QtCore/QtCore>
#include "QEthereum.h"

using namespace std;

QWebThree::QWebThree(QObject* _p): QObject(_p)
{
	moveToThread(_p->thread());
}

QWebThree::~QWebThree()
{
}

static QString toJsonRpcMessage(QString _json)
{
	QJsonObject f = QJsonDocument::fromJson(_json.toUtf8()).object();
	QJsonObject res;
	
	res["jsonrpc"] = "2.0";
	if (f.contains("call"))
		res["method"] = f["call"];
	if (f.contains("args"))
		res["params"] = f["args"];
	if (f.contains("_id"))
		res["id"] = f["_id"];
	
	return QString::fromUtf8(QJsonDocument(res).toJson());
}

static QString formatResponse(QString _json)
{
	QJsonObject f = QJsonDocument::fromJson(_json.toUtf8()).object();
	QJsonObject res;
	if (f.contains("id"))
		res["_id"] = f["id"];
	if (f.contains("result"))
		res["data"] = f["result"];
	
	return QString::fromUtf8(QJsonDocument(res).toJson());
}

void QWebThree::postData(QString _json)
{
	emit processData(toJsonRpcMessage(_json));
}

QWebThreeConnector::QWebThreeConnector(QWebThree* _q): m_qweb(_q)
{
}

QWebThreeConnector::~QWebThreeConnector()
{
	StopListening();
}

bool QWebThreeConnector::StartListening()
{
	connect(m_qweb, SIGNAL(processData(QString)), this, SLOT(onMessage(QString)));
	return true;
}

bool QWebThreeConnector::StopListening()
{
	this->disconnect();
	return true;
}

bool QWebThreeConnector::SendResponse(std::string const& _response, void* _addInfo)
{
	Q_UNUSED(_addInfo);
	emit m_qweb->send(formatResponse(QString::fromStdString(_response)));
	return true;
}

void QWebThreeConnector::onMessage(QString const& _json)
{
	OnRequest(_json.toStdString());
}


// extra bits needed to link on VS
#ifdef _MSC_VER

// include moc file, ofuscated to hide from automoc
#include\
"moc_QEthereum.cpp"

#endif
