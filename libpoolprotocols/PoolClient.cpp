#include "PoolClient.h"

using namespace std;
using namespace dev;
using namespace eth;

void PoolClient::setConnection(string const & host, string const & port, string const & user, string const & pass)
{
	m_host = host;
	m_port = port;
	m_user = user;
	m_pass = pass;
	m_connection_changed = true;
}
