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
/** @file CorsHttpServer.cpp
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2014
 */

#include "CorsHttpServer.h"

namespace jsonrpc
{

bool CorsHttpServer::SendResponse(std::string const& _response, void* _addInfo)
{
	struct mg_connection* conn = (struct mg_connection*) _addInfo;
	if (mg_printf(conn, "HTTP/1.1 200 OK\r\n"
				  "Content-Type: application/json\r\n"
				  "Content-Length: %d\r\n"
				  "Access-Control-Allow-Origin: *\r\n"
				  "Access-Control-Allow-Headers: Content-Type\r\n"
				  "\r\n"
				  "%s",(int)_response.length(), _response.c_str()) > 0)
		return true;
	return false;

}

}
