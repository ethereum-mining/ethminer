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
/** @file DappHost.cpp
 * @author Arkadiy Paronyan <arkadiy@ethdev.org>
 * @date 2015
 */

#include "DappHost.h"
#include <QUrl>
#include <microhttpd.h>
#include <boost/algorithm/string.hpp>
#include <libdevcore/Common.h>

using namespace dev;

DappHost::DappHost(int _port, int _threads):
	m_port(_port),
	m_url(QString("http://localhost:%1/").arg(m_port)),
	m_threads(_threads),
	m_running(false),
	m_daemon(nullptr)
{
	startListening();
}

DappHost::~DappHost()
{
	stopListening();
}

void DappHost::startListening()
{
	if(!this->m_running)
	{
		this->m_daemon = MHD_start_daemon(MHD_USE_SELECT_INTERNALLY, this->m_port, nullptr, nullptr, &DappHost::callback, this, MHD_OPTION_THREAD_POOL_SIZE, this->m_threads, MHD_OPTION_END);
		if (this->m_daemon != nullptr)
			this->m_running = true;
	}
}

void DappHost::stopListening()
{
	if(this->m_running)
	{
		MHD_stop_daemon(this->m_daemon);
		this->m_running = false;
	}
}

void DappHost::sendOptionsResponse(MHD_Connection* _connection)
{
	MHD_Response *result = MHD_create_response_from_data(0, NULL, 0, 1);
	MHD_add_response_header(result, "Allow", "GET, OPTIONS");
	MHD_add_response_header(result, "Access-Control-Allow-Headers", "origin, content-type, accept");
	MHD_add_response_header(result, "DAV", "1");
	MHD_queue_response(_connection, MHD_HTTP_OK, result);
	MHD_destroy_response(result);
}

void DappHost::sendNotAllowedResponse(MHD_Connection* _connection)
{
	MHD_Response *result = MHD_create_response_from_data(0, NULL, 0, 1);
	MHD_add_response_header(result, "Allow", "GET, OPTIONS");
	MHD_queue_response(_connection, MHD_HTTP_METHOD_NOT_ALLOWED, result);
	MHD_destroy_response(result);
}

void DappHost::sendResponse(std::string const& _url, MHD_Connection* _connection)
{
	QUrl requestUrl(QString::fromStdString(_url));
	QString path = requestUrl.path().toLower();
	if (path.isEmpty())
		path = "/";

	bytesConstRef response;
	unsigned code = MHD_HTTP_NOT_FOUND;
	std::string contentType;

	while (!path.isEmpty())
	{
		auto iter = m_entriesByPath.find(path);
		if (iter != m_entriesByPath.end())
		{
			ManifestEntry const* entry = iter->second;
			auto contentIter = m_dapp.content.find(entry->hash);
			if (contentIter == m_dapp.content.end())
				break;

			response = bytesConstRef(contentIter->second.data(), contentIter->second.size());
			code =  entry->httpStatus != 0 ? entry->httpStatus : MHD_HTTP_OK;
			contentType = entry->contentType;
			break;
		}
		path.truncate(path.length() - 1);
		path = path.mid(0, path.lastIndexOf('/'));
	}

	MHD_Response *result = MHD_create_response_from_data(response.size(), const_cast<byte*>(response.data()), 0, 1);
	if (!contentType.empty())
		MHD_add_response_header(result, "Content-Type", contentType.c_str());
	MHD_queue_response(_connection, code, result);
	MHD_destroy_response(result);
}

int DappHost::callback(void* _cls, MHD_Connection* _connection, char const* _url, char const* _method, char const* _version, char const* _uploadData, size_t* _uploadDataSize, void** _conCls)
{
	(void)_version;
	(void)_uploadData;
	(void)_uploadDataSize;
	(void)_conCls;
	DappHost* host = static_cast<DappHost*>(_cls);
	if (std::string("GET") == _method)
		host->sendResponse(std::string(_url), _connection);
	else if (std::string("OPTIONS") == _method)
		host->sendOptionsResponse(_connection);
	else
		host->sendNotAllowedResponse(_connection);
	return MHD_YES;
}

QUrl DappHost::hostDapp(Dapp&& _dapp)
{
	m_dapp = std::move(_dapp);
	m_entriesByPath.clear();
	for (ManifestEntry const& entry: m_dapp.manifest.entries)
		m_entriesByPath[QString::fromStdString(entry.path)] = &entry;

	return m_url;
}

bool DappHost::servesUrl(QUrl const& _url) const
{
	return m_url == _url || m_url.isParentOf(_url);
}
