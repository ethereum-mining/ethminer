/*************************************************************************
 * libjson-rpc-cpp
 *************************************************************************
 * @file    unixdomainsocketserver.h
 * @date    07.05.2015
 * @author  Alexandre Poirot <alexandre.poirot@legrand.fr>
 * @license See attached LICENSE.txt
 ************************************************************************/

#ifndef JSONRPC_CPP_UNIXDOMAINSOCKETSERVERCONNECTOR_H_
#define JSONRPC_CPP_UNIXDOMAINSOCKETSERVERCONNECTOR_H_

#include <stdarg.h>
#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <pthread.h>

#include <jsonrpccpp/server/abstractserverconnector.h>

namespace jsonrpc
{
	/**
	 * This class provides an embedded Unix Domain Socket Server,to handle incoming Requests.
	 */
	class UnixDomainSocketServer: public AbstractServerConnector
	{
		public:
			/**
			 * @brief UnixDomainSocketServer, constructor for the included UnixDomainSocketServer
			 * @param socket_path, a string containing the path to the unix socket
			 */
			UnixDomainSocketServer(const std::string& socket_path);

			virtual bool StartListening();
			virtual bool StopListening();

			bool virtual SendResponse(const std::string& response, void* addInfo = NULL);

		private:
			bool running;
			std::string socket_path;
			int socket_fd;
			struct sockaddr_un address;

			pthread_t listenning_thread;

			static void* LaunchLoop(void *p_data);
			void ListenLoop();
			struct ClientConnection
			{
				UnixDomainSocketServer *instance;
				int connection_fd;
			};
			static void* GenerateResponse(void *p_data);
			bool WriteToSocket(int fd, const std::string& toSend);
	};

} /* namespace jsonrpc */
#endif /* JSONRPC_CPP_HTTPSERVERCONNECTOR_H_ */

