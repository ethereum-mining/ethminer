
#include <jsonrpc/connectors/httpserver.h>

namespace jsonrpc
{

class CorsHttpServer : public HttpServer
{
public:
    using HttpServer::HttpServer;
    bool virtual SendResponse(const std::string& response,
            void* addInfo = NULL);
};

}

