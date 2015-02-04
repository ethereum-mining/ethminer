#include "ExecStats.h"

namespace dev
{
namespace eth
{
namespace jit
{

void ExecStats::execStarted()
{
	m_tp = std::chrono::high_resolution_clock::now();
}

void ExecStats::execEnded()
{
	execTime = std::chrono::high_resolution_clock::now() - m_tp;
}

}
}
}
