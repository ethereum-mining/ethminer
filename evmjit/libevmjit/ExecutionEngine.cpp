#include "ExecutionEngine.h"

#include <array>
#include <mutex>
#include <iostream>
#include <unordered_map>
#include <cstdlib>
#include <cstring>

#include "preprocessor/llvm_includes_start.h"
#include <llvm/IR/Module.h>
#include <llvm/ADT/Triple.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ManagedStatic.h>
#include "preprocessor/llvm_includes_end.h"

#include "evmjit/JIT.h"
#include "Runtime.h"
#include "Compiler.h"
#include "Optimizer.h"
#include "Cache.h"
#include "ExecStats.h"
#include "Utils.h"
#include "BuildInfo.gen.h"

namespace dev
{
namespace eth
{
namespace jit
{
using evmjit::JIT;

namespace
{
using EntryFuncPtr = ReturnCode(*)(Runtime*);

std::string codeHash(i256 const& _hash)
{
	static const auto size = sizeof(_hash);
	static const auto hexChars = "0123456789abcdef";
	std::string str;
	str.resize(size * 2);
	auto outIt = str.rbegin(); // reverse for BE
	auto& arr = *(std::array<byte, size>*)&_hash;
	for (auto b : arr)
	{
		*(outIt++) = hexChars[b & 0xf];
		*(outIt++) = hexChars[b >> 4];
	}
	return str;
}

void printVersion()
{
	std::cout << "Ethereum EVM JIT Compiler (http://github.com/ethereum/evmjit):\n"
			  << "  EVMJIT version " << EVMJIT_VERSION << "\n"
#ifdef NDEBUG
			  << "  Optimized build, " EVMJIT_VERSION_FULL "\n"
#else
			  << "  DEBUG build, " EVMJIT_VERSION_FULL "\n"
#endif
			  << "  Built " << __DATE__ << " (" << __TIME__ << ")\n"
			  << std::endl;
}

namespace cl = llvm::cl;
cl::opt<bool> g_optimize{"O", cl::desc{"Optimize"}};
cl::opt<CacheMode> g_cache{"cache", cl::desc{"Cache compiled EVM code on disk"},
	cl::values(
		clEnumValN(CacheMode::on,    "1", "Enabled"),
		clEnumValN(CacheMode::off,   "0", "Disabled"),
		clEnumValN(CacheMode::read,  "r", "Read only. No new objects are added to cache."),
		clEnumValN(CacheMode::write, "w", "Write only. No objects are loaded from cache."),
		clEnumValN(CacheMode::clear, "c", "Clear the cache storage. Cache is disabled."),
		clEnumValN(CacheMode::preload, "p", "Preload all cached objects."),
		clEnumValEnd)};
cl::opt<bool> g_stats{"st", cl::desc{"Statistics"}};
cl::opt<bool> g_dump{"dump", cl::desc{"Dump LLVM IR module"}};

void parseOptions()
{
	static llvm::llvm_shutdown_obj shutdownObj{};
	cl::AddExtraVersionPrinter(printVersion);
	//cl::ParseEnvironmentOptions("evmjit", "EVMJIT", "Ethereum EVM JIT Compiler");

	// FIXME: LLVM workaround:
	// Manually select instruction scheduler. Confirmed bad schedulers: source, list-burr, list-hybrid.
	// "source" scheduler has a bug: http://llvm.org/bugs/show_bug.cgi?id=22304
	auto envLine = std::getenv("EVMJIT");
	auto commandLine = std::string{"evmjit "} + (envLine ? envLine : "") + " -pre-RA-sched=list-ilp\0";
	static const auto c_maxArgs = 20;
	char const* argv[c_maxArgs] = {nullptr, };
	auto arg = std::strtok(&*commandLine.begin(), " ");
	auto i = 0;
	for (; i < c_maxArgs && arg; ++i, arg = std::strtok(nullptr, " "))
		argv[i] = arg;
	cl::ParseCommandLineOptions(i, argv, "Ethereum EVM JIT Compiler");
}

}


ReturnCode ExecutionEngine::run(RuntimeData* _data, Env* _env)
{
	static std::once_flag flag;
	std::call_once(flag, parseOptions);

	std::unique_ptr<ExecStats> listener{new ExecStats};
	listener->stateChanged(ExecState::Started);

	bool preloadCache = g_cache == CacheMode::preload;
	if (preloadCache)
		g_cache = CacheMode::on;

	// TODO: Do not pseudo-init the cache every time
	auto objectCache = (g_cache != CacheMode::off && g_cache != CacheMode::clear) ? Cache::getObjectCache(g_cache, listener.get()) : nullptr;

	static std::unique_ptr<llvm::ExecutionEngine> ee;
	if (!ee)
	{
		if (g_cache == CacheMode::clear)
			Cache::clear();

		llvm::InitializeNativeTarget();
		llvm::InitializeNativeTargetAsmPrinter();

		auto module = std::unique_ptr<llvm::Module>(new llvm::Module({}, llvm::getGlobalContext()));
		llvm::EngineBuilder builder(module.get());
		builder.setEngineKind(llvm::EngineKind::JIT);
		builder.setUseMCJIT(true);
		builder.setOptLevel(g_optimize ? llvm::CodeGenOpt::Default : llvm::CodeGenOpt::None);

		auto triple = llvm::Triple(llvm::sys::getProcessTriple());
		if (triple.getOS() == llvm::Triple::OSType::Win32)
			triple.setObjectFormat(llvm::Triple::ObjectFormatType::ELF);  // MCJIT does not support COFF format
		module->setTargetTriple(triple.str());

		ee.reset(builder.create());
		if (!CHECK(ee))
			return ReturnCode::LLVMConfigError;
		module.release();  // Successfully created llvm::ExecutionEngine takes ownership of the module
		ee->setObjectCache(objectCache);

		// FIXME: Disabled during API changes
		//if (preloadCache)
		//	Cache::preload(*ee, funcCache);
	}

	static StatsCollector statsCollector;

	auto mainFuncName = codeHash(_data->codeHash);
	m_runtime.init(_data, _env);

	// TODO: Remove cast
	auto entryFuncPtr = (EntryFuncPtr) JIT::getCode(_data->codeHash);
	if (!entryFuncPtr)
	{
		auto module = objectCache ? Cache::getObject(mainFuncName) : nullptr;
		if (!module)
		{
			listener->stateChanged(ExecState::Compilation);
			assert(_data->code || !_data->codeSize); //TODO: Is it good idea to execute empty code?
			module = Compiler{{}}.compile(_data->code, _data->code + _data->codeSize, mainFuncName);

			if (g_optimize)
			{
				listener->stateChanged(ExecState::Optimization);
				optimize(*module);
			}
		}
		if (g_dump)
			module->dump();

		ee->addModule(module.get());
		module.release();
		listener->stateChanged(ExecState::CodeGen);
		entryFuncPtr = (EntryFuncPtr)ee->getFunctionAddress(mainFuncName);
		if (!CHECK(entryFuncPtr))
			return ReturnCode::LLVMLinkError;
		JIT::mapCode(_data->codeHash, (uint64_t)entryFuncPtr); // FIXME: Remove cast
	}

	listener->stateChanged(ExecState::Execution);
	auto returnCode = entryFuncPtr(&m_runtime);
	listener->stateChanged(ExecState::Return);

	if (returnCode == ReturnCode::Return)
		returnData = m_runtime.getReturnData();     // Save reference to return data

	listener->stateChanged(ExecState::Finished);

	if (g_stats)
		statsCollector.stats.push_back(std::move(listener));

	return returnCode;
}

}
}
}
