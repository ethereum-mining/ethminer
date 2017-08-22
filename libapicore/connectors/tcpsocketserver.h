/*************************************************************************
 * libjson-rpc-cpp
 *************************************************************************
 * @file    tcpsocketserver.h
 * @date    17.07.2015
 * @author  Alexandre Poirot <alexandre.poirot@legrand.fr>
 * @license See attached LICENSE.txt
 ************************************************************************/

#ifndef JSONRPC_CPP_TCPSOCKETSERVERCONNECTOR_H_
#define JSONRPC_CPP_TCPSOCKETSERVERCONNECTOR_H_

#include <jsonrpccpp/server/abstractserverconnector.h>

namespace jsonrpc
{
	/**
         * This class provides an embedded TCP Socket Server that handle incoming Requests and send result over same socket.
         * It uses the delimiter character to distinct a full RPC Message over the tcp flow. This character is parametered on
         * compilation time in implementation files. The default value for this delimiter is 0x0A a.k.a. "new line".
         * This class hides OS specific features in real implementation of this server. Currently it has implementation for
         * both Linux and Windows.
         */
	class TcpSocketServer: public AbstractServerConnector
	{
		public:
			/**
                         * @brief TcpSocketServer, constructor for the included TcpSocketServer
                         * 
                         * Instanciates the real implementation of TcpSocketServerPrivate depending on running OS.
                         * 
                         * @param ipToBind The ipv4 address on which the server should bind and listen
                         * @param port The port on which the server should bind and listen
                         */
			TcpSocketServer(const std::string& ipToBind, const unsigned int &port);
			/**
                         * @brief ~TcpSocketServer, the destructor of TcpSocketServer
                         */
			~TcpSocketServer();
                        /**
                         * @brief The AbstractServerConnector::StartListening method overload.
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
                         * @brief The AbstractServerConnector::StopListening method overload.
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
                         * @brief The AbstractServerConnector::SendResponse method overload.
                         * 
                         * This method sends the result of the RPC Call over the tcp socket that the client has used to perform its request.
                         * @param response The response to send to the client
                         * @param addInfo Additionnal parameters (mainly client socket file descriptor)
                         * @return A boolean that indicates the success or the failure of the operation.
                         */
			bool SendResponse(const std::string& response, void* addInfo = NULL);

		private:
            AbstractServerConnector *realSocket; /*!< A pointer to the real implementation of this class depending of running OS*/
	};

} /* namespace jsonrpc */
#endif /* JSONRPC_CPP_TCPSOCKETSERVERCONNECTOR_H_ */

