/*************************************************************************
 * libjson-rpc-cpp
 *************************************************************************
 * @file    windowstcpsocketserver.h
 * @date    17.07.2015
 * @author  Alexandre Poirot <alexandre.poirot@legrand.fr>
 * @license See attached LICENSE.txt
 ************************************************************************/

#ifndef JSONRPC_CPP_WINDOWSTCPSOCKETSERVERCONNECTOR_H_
#define JSONRPC_CPP_WINDOWSTCPSOCKETSERVERCONNECTOR_H_

#include <stdarg.h>
#include <stdint.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#include <jsonrpccpp/server/abstractserverconnector.h>

namespace jsonrpc
{
	/**
         * This class is the Windows implementation of TCPSocketServer.
         * It uses the Winsock2 socket API and Windows thread API to performs its job.
         * Each client request is handled in a new thread.
         */
        class WindowsTcpSocketServer: public AbstractServerConnector
	{
		public:
			/**
                         * @brief WindowsTcpSocketServer, constructor of the Windows implementation of class TcpSocketServer
                         * @param ipToBind The ipv4 address on which the server should bind and listen
                         * @param port The port on which the server should bind and listen
                         */
			WindowsTcpSocketServer(const std::string& ipToBind, const unsigned int &port);
                        /**
                         * @brief ~WindowsTcpSocketServer, the destructor of WindowsTcpSocketServer
                         */
			~WindowsTcpSocketServer();
                        /**
                         * @brief The real implementation TcpSocketServer::StartListening method.
                         * 
                         * This method launches the listening loop that will handle client connections. 
                         * The return value depends on the current listening states :
                         *  - not listening and no error come up while bind and listen returns true
                         *  - not listening but error happen on bind or listen returns false
                         *  - is called while listening returns false
                         * 
                         * @return A boolean that indicates the success or the failure of the operation.
                         */
			bool StartListening();
                        /**
                         * @brief The real implementation TcpSocketServer::StopListening method.
                         * 
                         * This method stops the listening loop that will handle client connections. 
                         * The return value depends on the current listening states :
                         *  - listening and successfuly stops the listen loop returns true
                         *  - is called while not listening returns false
                         * 
                         * @return A boolean that indicates the success or the failure of the operation.
                         */
			bool StopListening();

                        /**
                         * @brief The real implementation TcpSocketServer::SendResponse method.
                         * 
                         * This method sends the result of the RPC Call over the tcp socket that the client has used to perform its request.
                         * @param response The response to send to the client
                         * @param addInfo Additionnal parameters (mainly client socket file descriptor)
                         * @return A boolean that indicates the success or the failure of the operation.
                         */
			bool SendResponse(const std::string& response, void* addInfo = NULL);

		private:
			bool running;               /*!< A boolean that is used to know the listening state*/
			std::string ipToBind;       /*!< The ipv4 address on which the server should bind and listen*/
			unsigned int port;          /*!< The port on which the server should bind and listen*/
			SOCKET socket_fd;           /*!< The file descriptior of the listening socket*/
			SOCKADDR_IN address;        /*!< The listening socket*/

			DWORD listenning_thread;    /*!< The identifier of the listen loop thread*/

                        /**
                         * @brief The static method that is used as listening thread entry point
                         * @param lp_data The parameters for the thread entry point method
                         */
			static DWORD WINAPI LaunchLoop(LPVOID lp_data);
                        /**
                         * @brief The method that launches the listenning loop
                         */
			void ListenLoop();
			struct GenerateResponseParameters
			{
				WindowsTcpSocketServer *instance;
				SOCKET connection_fd;
			};  /*!< The structure used to give parameters to the Response generating method*/
			/**
                         * @brief The static method that is used as client request handling entry point
                         * @param lp_data The parameters for the thread entry point method
                         */
                        static DWORD WINAPI GenerateResponse(LPVOID lp_data);
                        /**
                         * @brief A method that write a message to  socket
                         * 
                         * Tries to send the full message.
                         * @param fd The file descriptor of the socket message should be sent
                         * @param toSend The message to send over socket
                         * @returns A boolean indicating the success or the failure of the operation
                         */
			bool WriteToSocket(const SOCKET& fd, const std::string& toSend);
                        /**
                         * @brief A method that wait for the client to close the tcp session
                         * 
                         * This method wait for the client to close the tcp session in order to avoid the server to enter in TIME_WAIT status.
                         * Entering in TIME_WAIT status with too many clients may occur in a DOS attack 
                         * since server will not be able to use a new socket when a new client connects.
                         * @param fd The file descriptor of the socket that should be closed by the client
                         * @param timeout The maximum time the server will wait for the client to close the tcp session in milliseconds.
                         * @returns A boolean indicating the success or the failure of the operation
                         */
			bool WaitClientClose(const SOCKET& fd, const int &timeout = 100);
                        /**
                         * @brief A method that close a socket by reseting it
                         * 
                         * This method reset the tcp session in order to avoid enter in TIME_WAIT state.
                         * @param fd The file descriptor of the socket that should be reset
                         * @returns The return value of POSIX close() method
                         */
			int CloseByReset(const SOCKET& fd);
                        /**
                         * @brief A method that cleanly close a socket by avoid TIME_WAIT state
                         * 
                         * This method uses WaitClientClose and ClodeByReset to clenly close a tcp session with a client
                         * (avoiding TIME_WAIT to avoid DOS attacks).
                         * @param fd The file descriptor of the socket that should be cleanly closed
                         * @returns The return value of POSIX close() method
                         */
			int CleanClose(const SOCKET& fd);
	};

} /* namespace jsonrpc */
#endif /* JSONRPC_CPP_WINDOWSTCPSOCKETSERVERCONNECTOR_H_ */

