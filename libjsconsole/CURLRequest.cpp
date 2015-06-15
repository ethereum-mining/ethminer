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
/** @file CURLRequest.cpp
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 * Ethereum client.
 */

#include "CURLRequest.h"

using namespace std;

static size_t write_data(void *buffer, size_t elementSize, size_t numberOfElements, void *userp)
{
	static_cast<stringstream *>(userp)->write((const char *)buffer, elementSize * numberOfElements);
	return elementSize * numberOfElements;
}

void CURLRequest::commonCURLPreparation()
{
	m_resultBuffer.str("");
	curl_easy_setopt(m_curl, CURLOPT_URL, (m_url + "?").c_str());
	curl_easy_setopt(m_curl, CURLOPT_FOLLOWLOCATION, 1L);
	curl_easy_setopt(m_curl, CURLOPT_WRITEFUNCTION, write_data);
	curl_easy_setopt(m_curl, CURLOPT_WRITEDATA, &m_resultBuffer);
}

std::tuple<long, string> CURLRequest::commonCURLPerform()
{
	CURLcode res = curl_easy_perform(m_curl);
	if (res != CURLE_OK) {
		throw runtime_error(curl_easy_strerror(res));
	}
	long httpCode = 0;
	curl_easy_getinfo(m_curl, CURLINFO_RESPONSE_CODE, &httpCode);
	return make_tuple(httpCode, m_resultBuffer.str());
}

std::tuple<long, string> CURLRequest::post()
{
	commonCURLPreparation();
	curl_easy_setopt(m_curl, CURLOPT_POSTFIELDS, m_body.c_str());

	struct curl_slist *headerList = NULL;
	headerList = curl_slist_append(headerList, "Content-Type: application/json");
	curl_easy_setopt(m_curl, CURLOPT_HTTPHEADER, headerList);

	auto result = commonCURLPerform();

	curl_slist_free_all(headerList);
	return result;
}
