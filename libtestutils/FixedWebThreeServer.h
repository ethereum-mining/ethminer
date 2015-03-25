

#pragma once

#include <libdevcore/Exceptions.h>
#include <libweb3jsonrpc/WebThreeStubServerBase.h>

class FixedWebThreeServer: public dev::WebThreeStubServerBase, public dev::WebThreeStubDatabaseFace
{
public:
	FixedWebThreeServer(jsonrpc::AbstractServerConnector& _conn, std::vector<dev::KeyPair> const& _accounts, dev::eth::Interface* _client): WebThreeStubServerBase(_conn, _accounts), m_client(_client) {};

private:
	dev::eth::Interface* client() override { return m_client; }
	std::shared_ptr<dev::shh::Interface> face() override {	BOOST_THROW_EXCEPTION(dev::InterfaceNotSupported("dev::shh::Interface")); }
	dev::WebThreeNetworkFace* network() override { BOOST_THROW_EXCEPTION(dev::InterfaceNotSupported("dev::WebThreeNetworkFace")); }
	dev::WebThreeStubDatabaseFace* db() override { return this; }
	std::string get(std::string const& _name, std::string const& _key) override
	{
		std::string k(_name + "/" + _key);
		return m_db[k];
	}
	void put(std::string const& _name, std::string const& _key, std::string const& _value) override
	{
		std::string k(_name + "/" + _key);
		m_db[k] = _value;
	}

private:
	dev::eth::Interface* m_client;
	std::map<std::string, std::string> m_db;
};
