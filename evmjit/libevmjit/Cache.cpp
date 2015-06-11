#include "Cache.h"

#include <mutex>

#include "preprocessor/llvm_includes_start.h"
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Instructions.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_os_ostream.h>
#include "preprocessor/llvm_includes_end.h"

#include "ExecutionEngine.h"
#include "Utils.h"
#include "BuildInfo.gen.h"

namespace dev
{
namespace eth
{
namespace jit
{

namespace
{
	using Guard = std::lock_guard<std::mutex>;
	std::mutex x_cacheMutex;
	CacheMode g_mode;
	llvm::MemoryBuffer* g_lastObject;
	ExecutionEngineListener* g_listener;
	static const size_t c_versionStampLength = 32;

	llvm::StringRef getLibVersionStamp()
	{
		static auto version = llvm::SmallString<c_versionStampLength>{};
		if (version.empty())
		{
			version = EVMJIT_VERSION_FULL;
			version.resize(c_versionStampLength);
		}
		return version;
	}
}

ObjectCache* Cache::getObjectCache(CacheMode _mode, ExecutionEngineListener* _listener)
{
	static ObjectCache objectCache;

	Guard g{x_cacheMutex};

	g_mode = _mode;
	g_listener = _listener;
	return &objectCache;
}

void Cache::clear()
{
	Guard g{x_cacheMutex};

	using namespace llvm::sys;
	llvm::SmallString<256> cachePath;
	path::system_temp_directory(false, cachePath);
	path::append(cachePath, "evm_objs");

	std::error_code err;
	for (auto it = fs::directory_iterator{cachePath.str(), err}; it != fs::directory_iterator{}; it.increment(err))
		fs::remove(it->path());
}

void Cache::preload(llvm::ExecutionEngine& _ee, std::unordered_map<std::string, uint64_t>& _funcCache)
{
	Guard g{x_cacheMutex};

	// TODO: Cache dir should be in one place
	using namespace llvm::sys;
	llvm::SmallString<256> cachePath;
	path::system_temp_directory(false, cachePath);
	path::append(cachePath, "evm_objs");

	// Disable listener
	auto listener = g_listener;
	g_listener = nullptr;

	std::error_code err;
	for (auto it = fs::directory_iterator{cachePath.str(), err}; it != fs::directory_iterator{}; it.increment(err))
	{
		auto name = it->path().substr(cachePath.size() + 1);
		if (auto module = getObject(name))
		{
			DLOG(cache) << "Preload: " << name << "\n";
			_ee.addModule(module.get());
			module.release();
			auto addr = _ee.getFunctionAddress(name);
			assert(addr);
			_funcCache[std::move(name)] = addr;
		}
	}

	g_listener = listener;
}

std::unique_ptr<llvm::Module> Cache::getObject(std::string const& id)
{
	Guard g{x_cacheMutex};

	if (g_mode != CacheMode::on && g_mode != CacheMode::read)
		return nullptr;

	// TODO: Disabled because is not thread-safe.
	//if (g_listener)
	//	g_listener->stateChanged(ExecState::CacheLoad);

	DLOG(cache) << id << ": search\n";
	if (!CHECK(!g_lastObject))
		g_lastObject = nullptr;

	llvm::SmallString<256> cachePath;
	llvm::sys::path::system_temp_directory(false, cachePath);
	llvm::sys::path::append(cachePath, "evm_objs", id);

	if (auto r = llvm::MemoryBuffer::getFile(cachePath.str(), -1, false))
	{
		auto& buf = r.get();
		auto objVersionStamp = buf->getBufferSize() >= c_versionStampLength ? llvm::StringRef{buf->getBufferEnd() - c_versionStampLength, c_versionStampLength} : llvm::StringRef{};
		if (objVersionStamp == getLibVersionStamp())
			g_lastObject = llvm::MemoryBuffer::getMemBufferCopy(r.get()->getBuffer());
		else
			DLOG(cache) << "Unmatched version: " << objVersionStamp.str() << ", expected " << getLibVersionStamp().str() << "\n";
	}
	else if (r.getError() != std::make_error_code(std::errc::no_such_file_or_directory))
		DLOG(cache) << r.getError().message(); // TODO: Add warning log

	if (g_lastObject)  // if object found create fake module
	{
		DLOG(cache) << id << ": found\n";
		auto&& context = llvm::getGlobalContext();
		auto module = std::unique_ptr<llvm::Module>(new llvm::Module(id, context));
		auto mainFuncType = llvm::FunctionType::get(llvm::Type::getVoidTy(context), {}, false);
		auto mainFunc = llvm::Function::Create(mainFuncType, llvm::Function::ExternalLinkage, id, module.get());
		auto bb = llvm::BasicBlock::Create(context, {}, mainFunc);
		bb->getInstList().push_back(new llvm::UnreachableInst{context});
		return module;
	}
	DLOG(cache) << id << ": not found\n";
	return nullptr;
}


void ObjectCache::notifyObjectCompiled(llvm::Module const* _module, llvm::MemoryBuffer const* _object)
{
	Guard g{x_cacheMutex};

	// Only in "on" and "write" mode
	if (g_mode != CacheMode::on && g_mode != CacheMode::write)
		return;

	// TODO: Disabled because is not thread-safe.
	// if (g_listener)
		// g_listener->stateChanged(ExecState::CacheWrite);

	auto&& id = _module->getModuleIdentifier();
	llvm::SmallString<256> cachePath;
	llvm::sys::path::system_temp_directory(false, cachePath);
	llvm::sys::path::append(cachePath, "evm_objs");

	if (llvm::sys::fs::create_directory(cachePath.str()))
		DLOG(cache) << "Cannot create cache dir " << cachePath.str().str() << "\n";

	llvm::sys::path::append(cachePath, id);

	DLOG(cache) << id << ": write\n";
	std::string error;
	llvm::raw_fd_ostream cacheFile(cachePath.c_str(), error, llvm::sys::fs::F_None);
	cacheFile << _object->getBuffer() << getLibVersionStamp();
}

llvm::MemoryBuffer* ObjectCache::getObject(llvm::Module const* _module)
{
	Guard g{x_cacheMutex};

	DLOG(cache) << _module->getModuleIdentifier() << ": use\n";
	auto o = g_lastObject;
	g_lastObject = nullptr;
	return o;
}

}
}
}
