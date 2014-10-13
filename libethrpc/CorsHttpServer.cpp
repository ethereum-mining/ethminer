
#include "CorsHttpServer.h"

namespace jsonrpc
{

bool CorsHttpServer::SendResponse(const std::string &response, void *addInfo)
{
    struct mg_connection* conn = (struct mg_connection*) addInfo;
    if (mg_printf(conn, "HTTP/1.1 200 OK\r\n"
                  "Content-Type: application/json\r\n"
                  "Content-Length: %d\r\n"
                  "Access-Control-Allow-Origin: *\r\n"
                  "Access-Control-Allow-Headers: Content-Type\r\n"
                  "\r\n"
                  "%s",(int)response.length(), response.c_str()) > 0)


    {
        return true;
    }
    else
    {
        return false;
    }
}


}
