#pragma once

#include <string>
#include <chrono>

namespace dev
{
namespace eth
{
namespace jit
{

class ExecStats
{
public:
	std::string id;
	std::chrono::high_resolution_clock::duration compileTime;
	std::chrono::high_resolution_clock::duration codegenTime;
	std::chrono::high_resolution_clock::duration cacheLoadTime;
	std::chrono::high_resolution_clock::duration execTime;

	void execStarted();
	void execEnded();

private:
	std::chrono::high_resolution_clock::time_point m_tp;

};

}
}
}
