//
// Created by Marek Kotewicz on 15/06/15.
//

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
