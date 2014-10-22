#include <QtCore/QtCore>
#include <QtWebKitWidgets/QWebFrame>
#include <libdevcrypto/FileSystem.h>
#include <libevmface/Instruction.h>
#include <liblll/Compiler.h>
#include <libethereum/Client.h>
#include <libethereum/EthereumHost.h>
#include <libp2p/Host.h>
#include "QEthereum.h"

using namespace std;
using namespace dev;
using namespace dev::eth;


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
	emit m_qweb->send(QString::fromStdString(_response));
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
