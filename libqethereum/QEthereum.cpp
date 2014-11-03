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
	clientDieing();
}

static QString toJsonRpcBatch(std::vector<unsigned> const& _watches, QString _method)
{
	QJsonArray batch;
	for (int w: _watches)
	{
		QJsonObject res;
		res["jsonrpc"] = QString::fromStdString("2.0");
		res["method"] = _method;
		
		QJsonArray params;
		params.append(w);
		res["params"] = params;
		res["id"] = w;
		batch.append(res);
	}

	return QString::fromUtf8(QJsonDocument(batch).toJson());
}

void QWebThree::poll()
{
	if (m_watches.size() > 0)
	{
		QString batch = toJsonRpcBatch(m_watches, "changed");
		emit processData(batch, "changed");
	}
	if (m_shhWatches.size() > 0)
	{
		QString batch = toJsonRpcBatch(m_shhWatches, "shhChanged");
		emit processData(batch, "shhChanged");
	}
}

void QWebThree::clearWatches()
{
	if (m_watches.size() > 0)
	{
		QString batch = toJsonRpcBatch(m_watches, "uninstallFilter");
		m_watches.clear();
		emit processData(batch, "internal");
	}
	if (m_shhWatches.size() > 0)
	{
		QString batch = toJsonRpcBatch(m_shhWatches, "shhUninstallFilter");
		m_shhWatches.clear();
		emit processData(batch, "internal");
	}
}

void QWebThree::clientDieing()
{
	clearWatches();
	this->disconnect();
}

static QString formatInput(QJsonObject const& _object)
{
	QJsonObject res;
	res["jsonrpc"] = QString::fromStdString("2.0");
	res["method"] = _object["call"];
	res["params"] = _object["args"];
	res["id"] = _object["_id"];
	return QString::fromUtf8(QJsonDocument(res).toJson());
}

void QWebThree::postMessage(QString _json)
{
	QJsonObject f = QJsonDocument::fromJson(_json.toUtf8()).object();

	QString method = f["call"].toString();
	if (!method.compare("uninstallFilter") && f["args"].isArray() && f["args"].toArray().size())
	{
		int idToRemove = f["args"].toArray()[0].toInt();
		m_watches.erase(std::remove(m_watches.begin(), m_watches.end(), idToRemove), m_watches.end());
	}
	
	emit processData(formatInput(f), method);
}

static QString formatOutput(QJsonObject const& _object)
{
	QJsonObject res;
	res["_id"] = _object["id"];
	res["data"] = _object["result"];
	return QString::fromUtf8(QJsonDocument(res).toJson());
}

void QWebThree::onDataProcessed(QString _json, QString _addInfo)
{
	if (!_addInfo.compare("internal"))
		return;

	if (!_addInfo.compare("changed"))
	{
		QJsonArray resultsArray = QJsonDocument::fromJson(_json.toUtf8()).array();
		for (int i = 0; i < resultsArray.size(); i++)
		{
			QJsonObject elem = resultsArray[i].toObject();
			if (elem.contains("result") && elem["result"].toBool() == true)
			{
				QJsonObject res;
				res["_event"] = _addInfo;
				res["_id"] = (int)m_watches[i]; // we can do that couse poll is synchronous
				response(QString::fromUtf8(QJsonDocument(res).toJson()));
			}
		}
		return;
	}
	
	if (!_addInfo.compare("shhChanged"))
	{
		QJsonArray resultsArray = QJsonDocument::fromJson(_json.toUtf8()).array();
		for (int i = 0; i < resultsArray.size(); i++)
		{
			QJsonObject elem = resultsArray[i].toObject();
			if (elem.contains("result"))
			{
				QJsonObject res;
				res["_event"] = _addInfo;
				res["_id"] = (int)m_shhWatches[i];
				res["data"] = elem["result"].toArray();
				response(QString::fromUtf8(QJsonDocument(res).toJson()));
			}
		}
	}
	
	QJsonObject f = QJsonDocument::fromJson(_json.toUtf8()).object();
	
	if ((!_addInfo.compare("newFilter") || !_addInfo.compare("newFilterString")) && f.contains("result"))
		m_watches.push_back(f["result"].toInt());
	if (!_addInfo.compare("shhNewFilter") && f.contains("result"))
		m_shhWatches.push_back(f["result"].toInt());
	if (!_addInfo.compare("newIdentity") && f.contains("result"))
		emit onNewId(f["result"].toString());

	response(formatOutput(f));
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

// extra bits needed to link on VS
#ifdef _MSC_VER

// include moc file, ofuscated to hide from automoc
#include\
"moc_QEthereum.cpp"

#endif
