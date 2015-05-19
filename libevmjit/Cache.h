#pragma once

#include <memory>
#include <unordered_map>

#include "preprocessor/llvm_includes_start.h"
#include <llvm/ExecutionEngine/ObjectCache.h>
#include "preprocessor/llvm_includes_end.h"

namespace llvm
{
	class ExecutionEngine;
}

namespace dev
{
namespace evmjit
{
class JITListener;

enum class CacheMode
{
	on,
	off,
	read,
	write,
	clear,
	preload
};

class ObjectCache : public llvm::ObjectCache
{
public:
	/// notifyObjectCompiled - Provides a pointer to compiled code for Module M.
	virtual void notifyObjectCompiled(llvm::Module const* _module, llvm::MemoryBufferRef _object) final override;

	/// getObjectCopy - Returns a pointer to a newly allocated MemoryBuffer that
	/// contains the object which corresponds with Module M, or 0 if an object is
	/// not available. The caller owns both the MemoryBuffer returned by this
	/// and the memory it references.
	virtual std::unique_ptr<llvm::MemoryBuffer> getObject(llvm::Module const* _module) final override;
};


class Cache
{
public:
	static ObjectCache* init(CacheMode _mode, JITListener* _listener);
	static std::unique_ptr<llvm::Module> getObject(std::string const& id);

	/// Clears cache storage
	static void clear();

	/// Loads all available cached objects to ExecutionEngine
	static void preload(llvm::ExecutionEngine& _ee, std::unordered_map<std::string, uint64_t>& _funcCache);
};

}
}
