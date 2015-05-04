//
// Created by Marek Kotewicz on 28/04/15.
//

#pragma once

#include <libjsengine/JSV8Engine.h>
#include <libjsengine/JSV8Printer.h>
#include <libweb3jsonrpc/WebThreeStubServer.h>

namespace dev
{
namespace eth
{

class JSConsole
{
public:
	JSConsole(WebThreeDirect& _web3, std::vector<dev::KeyPair> const& _accounts);
	void repl() const;

private:
	std::string promptForIndentionLevel(int _i) const;

	JSV8Engine m_engine;
	JSV8Printer m_printer;
	std::unique_ptr<WebThreeStubServer> m_jsonrpcServer;
	std::unique_ptr<jsonrpc::AbstractServerConnector> m_jsonrpcConnector;
};

}
}
