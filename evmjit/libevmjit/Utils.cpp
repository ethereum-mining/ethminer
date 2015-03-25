#include "Utils.h"

#include <llvm/Support/Debug.h>

#include "BuildInfo.gen.h"

#if !defined(NDEBUG) // Debug

namespace dev
{
namespace evmjit
{

std::ostream& getLogStream(char const* _channel)
{
	static std::ostream nullStream{nullptr};
#if LLVM_DEBUG
	return (llvm::DebugFlag && llvm::isCurrentDebugType(_channel)) ? std::cerr : nullStream;
#else
	return (void)_channel, nullStream;
#endif
}

}
}

#endif
