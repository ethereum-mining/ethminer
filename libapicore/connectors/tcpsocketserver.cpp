/*************************************************************************
 * libjson-rpc-cpp
 *************************************************************************
 * @file    tcpsocketserver.cpp
 * @date    17.07.2015
 * @author  Alexandre Poirot <alexandre.poirot@legrand.fr>
 * @license See attached LICENSE.txt
 ************************************************************************/

#include "tcpsocketserver.h"
#ifdef WIN32
#include "windowstcpsocketserver.h"
#elif __unix__
#include "linuxtcpsocketserver.h"
#endif
#include <string>

#include <libdevcore/Log.h>

using namespace jsonrpc;
using namespace std;

TcpSocketServer::TcpSocketServer(const std::string& ipToBind, const unsigned int &port) :
	AbstractServerConnector()
{
#ifdef WIN32
	this->realSocket = new WindowsTcpSocketServer(ipToBind, port);
#elif __unix__
	this->realSocket = new LinuxTcpSocketServer(ipToBind, port);
#else
	this->realSocket = NULL;
#endif
}

TcpSocketServer::~TcpSocketServer()
{
	if(this->realSocket != NULL) 
	{
		delete this->realSocket;
		this->realSocket = NULL;
	}
}

bool TcpSocketServer::StartListening()
{
	if(this->realSocket != NULL)
	{
		this->realSocket->SetHandler(this->GetHandler());
		return this->realSocket->StartListening();
	}
	else 
		return false;
}

bool TcpSocketServer::StopListening()
{
	if(this->realSocket != NULL)
		return this->realSocket->StopListening();
	else
		return false;
}

bool TcpSocketServer::SendResponse(const string& response, void* addInfo)
{
	if(this->realSocket != NULL)
		return this->realSocket->SendResponse(response, addInfo);
	else
		return false;
}
