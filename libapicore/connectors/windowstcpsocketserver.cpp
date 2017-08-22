/*************************************************************************
 * libjson-rpc-cpp
 *************************************************************************
 * @file    windowstcpsocketserver.cpp
 * @date    17.07.2015
 * @author  Alexandre Poirot <alexandre.poirot@legrand.fr>
 * @license See attached LICENSE.txt
 ************************************************************************/

#include "windowstcpsocketserver.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <windows.h>

#include <sstream>
#include <iostream>
#include <string>
#include <libdevcore/Log.h>
#include <jsonrpccpp/common/specificationparser.h>

using namespace jsonrpc;
using namespace std;

#define BUFFER_SIZE 64
#ifndef DELIMITER_CHAR
#define DELIMITER_CHAR char(0x0A)
#endif //DELIMITER_CHAR

WindowsTcpSocketServer::WindowsTcpSocketServer(const std::string& ipToBind, const unsigned int &port) : AbstractServerConnector(), ipToBind(ipToBind), port(port), running(false)
{
}

WindowsTcpSocketServer::~WindowsTcpSocketServer()
{
}

bool WindowsTcpSocketServer::StartListening()
{
	if(!this->running)
	{
		//Create and bind socket here.
		//Then launch the listenning loop.
		this->socket_fd = socket(AF_INET, SOCK_STREAM, 0);
		if(this->socket_fd < 0)
		{
			return false;
		}
		unsigned long nonBlocking = 1;
		ioctlsocket(this->socket_fd, FIONBIO, &nonBlocking); //Set non blocking
                int reuseaddr = 1;
                setsockopt(this->socket_fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<char*>(&reuseaddr), sizeof(reuseaddr));

		/* start with a clean address structure */
		memset(&(this->address), 0, sizeof(SOCKADDR_IN));

		this->address.sin_family = AF_INET;
		this->address.sin_addr.s_addr = inet_addr(this->ipToBind.c_str());
		this->address.sin_port = htons(this->port);

		if(::bind(this->socket_fd, reinterpret_cast<SOCKADDR*>(&(this->address)), sizeof(SOCKADDR_IN)) != 0)
		{
			return false;
		}

		if(listen(this->socket_fd, 5) != 0)
		{
			return false;
		}
				
		//Launch listening loop there
		this->running = true;
		HANDLE ret = CreateThread(NULL, 0, reinterpret_cast<LPTHREAD_START_ROUTINE>(&(WindowsTcpSocketServer::LaunchLoop)), reinterpret_cast<LPVOID>(this), 0, &(this->listenning_thread));
		if(ret == NULL)
		{
			ExitProcess(3);
		}
		else
		{
			CloseHandle(ret);
		}
		this->running = static_cast<bool>(ret!=NULL);
		return this->running;
	}
	else
	{
		return false;
	}

}

bool WindowsTcpSocketServer::StopListening()
{
	if(this->running)
	{
		this->running = false;
		WaitForSingleObject(OpenThread(THREAD_ALL_ACCESS, FALSE,this->listenning_thread), INFINITE);
		closesocket(this->socket_fd);
		return !(this->running);
	}
	else
	{
		return false;
	}
}

bool WindowsTcpSocketServer::SendResponse(const string& response, void* addInfo)
{
	bool result = false;
	int connection_fd = reinterpret_cast<intptr_t>(addInfo);

	string temp = response;
	if(temp.find(DELIMITER_CHAR) == string::npos)
	{
		temp.append(1, DELIMITER_CHAR);
	}
	if(DELIMITER_CHAR != '\n')
	{
		char eot = DELIMITER_CHAR;
		string toSend = temp.substr(0, toSend.find_last_of('\n'));
		toSend += eot;
		result = this->WriteToSocket(connection_fd, toSend);
	}
	else
	{
		result = this->WriteToSocket(connection_fd, temp);
	}
	CleanClose(connection_fd);
	return result;
}

DWORD WINAPI WindowsTcpSocketServer::LaunchLoop(LPVOID lp_data)
{
	WindowsTcpSocketServer *instance = reinterpret_cast<WindowsTcpSocketServer*>(lp_data);;
	instance->ListenLoop();
	CloseHandle(GetCurrentThread());
	return 0; //DO NOT USE ExitThread function here! ExitThread does not call destructors for allocated objects and therefore it would lead to a memory leak.
}

void WindowsTcpSocketServer::ListenLoop()
{
	while(this->running)
	{
		SOCKET connection_fd = INVALID_SOCKET;
		SOCKADDR_IN connection_address;
		memset(&connection_address, 0, sizeof(SOCKADDR_IN));
		int address_length = sizeof(connection_address);
		if((connection_fd = accept(this->socket_fd, reinterpret_cast<SOCKADDR*>(&connection_address),  &address_length)) != INVALID_SOCKET)
		{
			unsigned long nonBlocking = 0;
			ioctlsocket(connection_fd, FIONBIO, &nonBlocking); //Set blocking
			DWORD client_thread;
			struct GenerateResponseParameters *params = new struct GenerateResponseParameters();
			params->instance = this;
			params->connection_fd = connection_fd;
			HANDLE ret = CreateThread(NULL, 0, reinterpret_cast<LPTHREAD_START_ROUTINE>(&(WindowsTcpSocketServer::GenerateResponse)), reinterpret_cast<LPVOID>(params), 0, &client_thread);
			if(ret == NULL)
			{
				delete params;
				params = NULL;
				CleanClose(connection_fd);
			}
			else
			{
				CloseHandle(ret);
			}
		}
		else
		{
			Sleep(2.5);
		}
	}
}

DWORD WINAPI WindowsTcpSocketServer::GenerateResponse(LPVOID lp_data)
{
	struct GenerateResponseParameters* params = reinterpret_cast<struct GenerateResponseParameters*>(lp_data);
	WindowsTcpSocketServer *instance = params->instance;
	int connection_fd = params->connection_fd;
	delete params;
	params = NULL;
	int nbytes = 0;
	char buffer[BUFFER_SIZE];
	memset(&buffer, 0, BUFFER_SIZE);
	string request = "";
	do
	{ //The client sends its json formatted request and a delimiter request.
		nbytes = recv(connection_fd, buffer, BUFFER_SIZE, 0);
		if(nbytes == -1)
		{
			instance->CleanClose(connection_fd);
		}
		else
		{
			request.append(buffer,nbytes);
		}
	} while(request.find(DELIMITER_CHAR) == string::npos);
	instance->OnRequest(request, reinterpret_cast<void*>(connection_fd));
	CloseHandle(GetCurrentThread());
	return 0; //DO NOT USE ExitThread function here! ExitThread does not call destructors for allocated objects and therefore it would lead to a memory leak.
}

bool WindowsTcpSocketServer::WriteToSocket(const SOCKET& fd, const string& toWrite)
{
	bool fullyWritten = false;
	bool errorOccured = false;
	string toSend = toWrite;
	do
	{
		unsigned long byteWritten = send(fd, toSend.c_str(), toSend.size(), 0);
		if(byteWritten < 0)
		{
			errorOccured = true;
			CleanClose(fd);
		}
		else if(byteWritten < toSend.size())
		{
			int len = toSend.size() - byteWritten;
			toSend = toSend.substr(byteWritten + sizeof(char), len);
		}
		else
			fullyWritten = true;
	} while(!fullyWritten && !errorOccured);

	return fullyWritten && !errorOccured;
}

bool WindowsTcpSocketServer::WaitClientClose(const SOCKET& fd, const int &timeout)
{
	bool ret = false;
	int i = 0;
	while((recv(fd, NULL, 0, 0) != 0) && i < timeout)
	{
		Sleep(1);
		++i;
		ret = true;
	}

	return ret;
}

int WindowsTcpSocketServer::CloseByReset(const SOCKET& fd)
{
	struct linger so_linger;
	so_linger.l_onoff = 1;
	so_linger.l_linger = 0;

	int ret = setsockopt(fd, SOL_SOCKET, SO_LINGER, reinterpret_cast<char*>(&so_linger), sizeof(so_linger));
	if(ret != 0)
		return ret;

	return closesocket(fd);
}

int WindowsTcpSocketServer::CleanClose(const SOCKET& fd)
{
	if(WaitClientClose(fd))
	{
		return closesocket(fd);
	}
	else
	{
		return CloseByReset(fd);
	}
}

//This is inspired from SFML to manage Winsock initialization. Thanks to them! ( http://www.sfml-dev.org/ ).
struct ServerSocketInitializer
{
	ServerSocketInitializer()
	{
		cwarn << "INIT!";
		WSADATA init;
		if(WSAStartup(MAKEWORD(2, 2), &init) != 0)
		{
                     JsonRpcException(Errors::ERROR_CLIENT_CONNECTOR, "An issue occured while WSAStartup executed.");
		}
	}

	~ServerSocketInitializer()
	{
		if(WSACleanup() != 0)
		{
                         cerr << "An issue occured while WSAClean executed." << endl;
		}
	}
};

struct ServerSocketInitializer serverGlobalInitializer;
