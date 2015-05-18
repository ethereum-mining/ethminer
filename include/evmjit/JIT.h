#pragma once

#include "evmjit/DataTypes.h"

namespace dev
{
namespace evmjit
{

class JIT
{
public:

	/// Ask JIT if the EVM code is ready for execution.
	/// Returns `true` if the EVM code has been compiled and loaded into memory.
	/// In this case the code can be executed without overhead.
	/// \param _codeHash	The Keccak hash of the EVM code.
	static bool isCodeReady(h256 _codeHash);
};

}
}
