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
/** @file CURLRequest.h
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 * Ethereum client.
 */

// based on http://stackoverflow.com/questions/1011339/how-do-you-make-a-http-request-with-c/27026683#27026683

#pragma once

#include <stdio.h>
#include <sstream>
#include <unordered_map>
#include <curl/curl.h>

class CURLRequest
{
public:
	CURLRequest(): m_curl(curl_easy_init()) {}
	~CURLRequest()
	{
		if (m_curl)
			curl_easy_cleanup(m_curl);
	}

	void setUrl(std::string _url) { m_url = _url; }
	void setBody(std::string _body) { m_body = _body; }

	std::tuple<long, std::string> post();

private:
	std::string m_url;
	std::string m_body;

	CURL* m_curl;
	std::stringstream m_resultBuffer;

	void commonCURLPreparation();
	std::tuple<long, std::string> commonCURLPerform();
};


