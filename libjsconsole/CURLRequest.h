//
// Created by Marek Kotewicz on 15/06/15.
//

// based on http://stackoverflow.com/questions/1011339/how-do-you-make-a-http-request-with-c/27026683#27026683

#include <stdio.h>
#include <sstream>
#include <unordered_map>
#include <curl/curl.h>

class CURLRequest
{
public:
	CURLRequest(): m_curl(curl_easy_init()) {};
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


