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

#include "ExecutionContext.h"
#include "evmjit/JIT.h"
#include "Compiler.h"
#include "Optimizer.h"
#include "Cache.h"
#include "ExecStats.h"
#include "Utils.h"
#include "BuildInfo.gen.h"

namespace dev
{
namespace evmjit
{

namespace
{
using EntryFuncPtr = ReturnCode(*)(ExecutionContext*);

std::string hash2str(i256 const& _hash)
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

void printVersion() // FIXME: Fix LLVM version parsing
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
	cl::ParseEnvironmentOptions("evmjit", "EVMJIT", "Ethereum EVM JIT Compiler");
}

std::unique_ptr<llvm::ExecutionEngine> init()
{
	/// ExecutionEngine is created only once

	parseOptions();

	bool preloadCache = g_cache == CacheMode::preload;
	if (preloadCache)
		g_cache = CacheMode::on;

	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	auto module = std::unique_ptr<llvm::Module>(new llvm::Module({}, llvm::getGlobalContext()));

	// FIXME: LLVM 3.7: test on Windows
	auto triple = llvm::Triple(llvm::sys::getProcessTriple());
	if (triple.getOS() == llvm::Triple::OSType::Win32)
		triple.setObjectFormat(llvm::Triple::ObjectFormatType::ELF);  // MCJIT does not support COFF format
	module->setTargetTriple(triple.str());

	llvm::EngineBuilder builder(std::move(module));
	builder.setEngineKind(llvm::EngineKind::JIT);
	builder.setOptLevel(g_optimize ? llvm::CodeGenOpt::Default : llvm::CodeGenOpt::None);

	auto ee = std::unique_ptr<llvm::ExecutionEngine>{builder.create()};

	// TODO: Update cache listener
	ee->setObjectCache(Cache::init(g_cache, nullptr));

	// FIXME: Disabled during API changes
	//if (preloadCache)
	//	Cache::preload(*ee, funcCache);

	return ee;
}

}

ReturnCode ExecutionEngine::run(ExecutionContext& _context)
{
	static auto s_ee = init();

	std::unique_ptr<ExecStats> listener{new ExecStats};
	listener->stateChanged(ExecState::Started);

	auto code = _context.code();
	auto codeSize = _context.codeSize();
	auto codeHash = _context.codeHash();

	static StatsCollector statsCollector;

	auto mainFuncName = hash2str(codeHash);

	// TODO: Remove cast
	auto entryFuncPtr = (EntryFuncPtr) JIT::getCode(codeHash);
	if (!entryFuncPtr)
	{
		auto module = Cache::getObject(mainFuncName);
		if (!module)
		{
			listener->stateChanged(ExecState::Compilation);
			assert(code || !codeSize); //TODO: Is it good idea to execute empty code?
			module = Compiler{{}}.compile(code, code + codeSize, mainFuncName);

			if (g_optimize)
			{
				listener->stateChanged(ExecState::Optimization);
				optimize(*module);
			}

			prepare(*module);
		}
		if (g_dump)
			module->dump();

		s_ee->addModule(std::move(module));
		listener->stateChanged(ExecState::CodeGen);
		entryFuncPtr = (EntryFuncPtr)s_ee->getFunctionAddress(mainFuncName);
		if (!CHECK(entryFuncPtr))
			return ReturnCode::LLVMLinkError;
		JIT::mapCode(codeHash, (void*)entryFuncPtr); // FIXME: Remove cast
	}

	listener->stateChanged(ExecState::Execution);
	auto returnCode = entryFuncPtr(&_context);
	listener->stateChanged(ExecState::Return);

	if (returnCode == ReturnCode::Return)
		_context.returnData = _context.getReturnData();     // Save reference to return data

	listener->stateChanged(ExecState::Finished);

	if (g_stats)
		statsCollector.stats.push_back(std::move(listener));

	return returnCode;
}

}
}
