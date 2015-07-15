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
/** @file DappHost.h
 * @author Arkadiy Paronyan <arkadiy@ethdev.org>
 * @date 2015
 */

#pragma once

#include <map>
#include <QUrl>
#include <QString>
#include "DappLoader.h"

struct MHD_Daemon;
struct MHD_Connection;

/// DApp web server. Servers web content, resolves paths by hashes
class DappHost
{
public:
	/// @param _port Network pork to listen for incoming connections
	/// @param _threads Max number of threads to process requests
	DappHost(int _port, int _threads = 10);
	virtual ~DappHost();
	/// Load and host a dapp. Previsous dapp in discarded. Synchronous
	QUrl hostDapp(Dapp&& _dapp);

	/// @returns true if the given url is served from this DappHost.
	bool servesUrl(QUrl const& _url) const;

private:
	void startListening();
	void stopListening();
	void sendOptionsResponse(MHD_Connection* _connection);
	void sendNotAllowedResponse(MHD_Connection* _connection);
	void sendResponse(std::string const& _url, MHD_Connection* _connection);
	static int callback(void* _cls, MHD_Connection* _connection, char const* _url, char const* _method, char const* _version, char const* _uploadData, size_t* _uploadDataSize, void** _conCls);

	int const m_port;
	QUrl const m_url;
	int m_threads;
	bool m_running;
	MHD_Daemon* m_daemon;
	Dapp m_dapp;
	std::map<QString, ManifestEntry const*> m_entriesByPath;
};

