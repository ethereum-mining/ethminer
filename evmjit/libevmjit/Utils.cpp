#include "Utils.h"

#include <llvm/Support/Debug.h>

#if !defined(NDEBUG) // Debug

namespace dev
{
namespace evmjit
{

std::ostream& getLogStream(char const* _channel)
{
	static std::ostream nullStream{nullptr};
	return (llvm::DebugFlag && llvm::isCurrentDebugType(_channel)) ? std::cerr : nullStream;
}

}
}

#endif
