/*************************************************************************
 * libjson-rpc-cpp
 *************************************************************************
 * @file    linuxtcpsocketserver.cpp
 * @date    17.07.2015
 * @author  Alexandre Poirot <alexandre.poirot@legrand.fr>
 * @license See attached LICENSE.txt
 ************************************************************************/

#include "linuxtcpsocketserver.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>

#include <sstream>
#include <iostream>
#include <string>

#include <jsonrpccpp/common/specificationparser.h>

#include <errno.h>

using namespace jsonrpc;
using namespace std;

#define BUFFER_SIZE 64
#ifndef DELIMITER_CHAR
#define DELIMITER_CHAR char(0x0A)
#endif //DELIMITER_CHAR

LinuxTcpSocketServer::LinuxTcpSocketServer(const std::string& ipToBind, const unsigned int &port) :
    AbstractServerConnector(),
    running(false),
	ipToBind(ipToBind),
    port(port)
{
}

bool LinuxTcpSocketServer::StartListening()
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

		fcntl(this->socket_fd, F_SETFL, FNDELAY);
		int reuseaddr = 1;
		setsockopt(this->socket_fd, SOL_SOCKET, SO_REUSEADDR, &reuseaddr, sizeof(reuseaddr));

		/* start with a clean address structure */
		memset(&(this->address), 0, sizeof(struct sockaddr_in));

		this->address.sin_family = AF_INET;
		inet_aton(this->ipToBind.c_str(), &(this->address.sin_addr));
		this->address.sin_port = htons(this->port);

		if(::bind(this->socket_fd, reinterpret_cast<struct sockaddr *>(&(this->address)), sizeof(struct sockaddr_in)) != 0)
		{
			return false;
		}

		if(listen(this->socket_fd, 5) != 0)
		{
			return false;
		}
		//Launch listening loop there
		this->running = true;
		int ret = pthread_create(&(this->listenning_thread), NULL, LinuxTcpSocketServer::LaunchLoop, this);
		if(ret != 0)
		{
			pthread_detach(this->listenning_thread);
			shutdown(this->socket_fd, 2);
			close(this->socket_fd);
		}
		this->running = static_cast<bool>(ret==0);
		return this->running;
	}
	else
	{
		return false;
	}
}

bool LinuxTcpSocketServer::StopListening()
{
	if(this->running)
	{
		this->running = false;
		pthread_join(this->listenning_thread, NULL);
		shutdown(this->socket_fd, 2);
		close(this->socket_fd);
		return !(this->running);
	}
	else
	{
		return false;
	}
}

bool LinuxTcpSocketServer::SendResponse(const string& response, void* addInfo)
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

void* LinuxTcpSocketServer::LaunchLoop(void *p_data)
{
	pthread_detach(pthread_self());
	LinuxTcpSocketServer *instance = reinterpret_cast<LinuxTcpSocketServer*>(p_data);;
	instance->ListenLoop();
	return NULL;
}

void LinuxTcpSocketServer::ListenLoop()
{
	int connection_fd = 0;
	struct sockaddr_in connection_address;
	memset(&connection_address, 0, sizeof(struct sockaddr_in));
	socklen_t address_length = sizeof(connection_address);
	while(this->running)
	{
		if((connection_fd = accept(this->socket_fd, reinterpret_cast<struct sockaddr *>(&(connection_address)),  &address_length)) > 0)
		{
			pthread_t client_thread;
			struct GenerateResponseParameters *params = new struct GenerateResponseParameters();
			params->instance = this;
			params->connection_fd = connection_fd;
			int ret = pthread_create(&client_thread, NULL, LinuxTcpSocketServer::GenerateResponse, params);
			if(ret != 0)
			{
				pthread_detach(client_thread);
				delete params;
				params = NULL;
				CleanClose(connection_fd);
			}
		}
		else
		{
			usleep(2500);
		}
	}
}

void* LinuxTcpSocketServer::GenerateResponse(void *p_data)
{
	pthread_detach(pthread_self());
	struct GenerateResponseParameters* params = reinterpret_cast<struct GenerateResponseParameters*>(p_data);
	LinuxTcpSocketServer *instance = params->instance;
	int connection_fd = params->connection_fd;
	delete params;
	params = NULL;
	int nbytes;
	char buffer[BUFFER_SIZE];
	string request;
	do
	{ //The client sends its json formatted request and a delimiter request.
		nbytes = recv(connection_fd, buffer, BUFFER_SIZE, 0);
		if(nbytes == -1)
		{
			instance->CleanClose(connection_fd);
			return NULL;
		}
		else
		{
			request.append(buffer,nbytes);
		}
	} while(request.find(DELIMITER_CHAR) == string::npos);
	instance->OnRequest(request, reinterpret_cast<void*>(connection_fd));
	return NULL;
}


bool LinuxTcpSocketServer::WriteToSocket(const int& fd, const string& toWrite)
{
	bool fullyWritten = false;
	bool errorOccured = false;
	string toSend = toWrite;
	do
	{
		ssize_t byteWritten = send(fd, toSend.c_str(), toSend.size(), 0);
		if(byteWritten < 0)
		{
			errorOccured = true;
			CleanClose(fd);
		}
		else if(static_cast<size_t>(byteWritten) < toSend.size())
		{
			int len = toSend.size() - byteWritten;
			toSend = toSend.substr(byteWritten + sizeof(char), len);
		}
		else
			fullyWritten = true;
	} while(!fullyWritten && !errorOccured);

	return fullyWritten && !errorOccured;
}

bool LinuxTcpSocketServer::WaitClientClose(const int& fd, const int &timeout)
{
	bool ret = false;
	int i = 0;
	while((recv(fd, NULL, 0, 0) != 0) && i < timeout)
	{
		usleep(1);
		++i;
		ret = true;
	}

	return ret;
}

int LinuxTcpSocketServer::CloseByReset(const int& fd)
{
	struct linger so_linger;
	so_linger.l_onoff = 1;
	so_linger.l_linger = 0;

	int ret = setsockopt(fd, SOL_SOCKET, SO_LINGER, &so_linger, sizeof(so_linger));
	if(ret != 0)
		return ret;

	return close(fd);
}

int LinuxTcpSocketServer::CleanClose(const int& fd)
{
	if(WaitClientClose(fd))
	{
		return close(fd);
	}
	else
	{
		return CloseByReset(fd);
	}
}
