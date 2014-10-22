#pragma once

#include <QtCore/QObject>
#include <QtCore/QStringList>
#include <QtCore/QList>
#include <libdevcore/CommonIO.h>
#include <libethcore/CommonEth.h>
#include <jsonrpc/rpc.h>

class QWebThree: public QObject
{
	Q_OBJECT
	
public:
	QWebThree(QObject* _p);
	virtual ~QWebThree();

	Q_INVOKABLE void postData(QString _json);
	
signals:
	void processData(QString _json);
	void send(QString _json);
};

class QWebThreeConnector: public QObject, public jsonrpc::AbstractServerConnector
{
	Q_OBJECT
	
public:
	QWebThreeConnector(QWebThree* _q);
	virtual ~QWebThreeConnector();
	
	virtual bool StartListening();
	virtual bool StopListening();
	
	bool virtual SendResponse(std::string const& _response,
							  void* _addInfo = NULL);
	
public slots:
	void onMessage(QString const& _json);
	
private:
	QWebThree* m_qweb;
};


// TODO: p2p object condition
#define QETH_INSTALL_JS_NAMESPACE(_frame, _env, qweb) [_frame, _env, qweb]() \
{ \
	_frame->disconnect(); \
	_frame->addToJavaScriptWindowObject("_web3", qweb, QWebFrame::ScriptOwnership); \
	_frame->evaluateJavaScript("navigator.qt = _web3;"); \
	_frame->evaluateJavaScript("(function () {" \
							"navigator.qt.handlers = [];" \
							"Object.defineProperty(navigator.qt, 'onmessage', {" \
							"	set: function(handler) {" \
							"		navigator.qt.handlers.push(handler);" \
							"	}" \
							"})" \
							"})()"); \
	_frame->evaluateJavaScript("navigator.qt.send.connect(function (res) {" \
							"navigator.qt.handlers.forEach(function (handler) {" \
							"	handler(res);" \
							"})" \	
							"})"); \
}


