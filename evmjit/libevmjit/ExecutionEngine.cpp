#include "ExecutionEngine.h"

#include <chrono>
#include <cstdlib>	// env options

#include <llvm/IR/Module.h>
#include <llvm/ADT/Triple.h>
#pragma warning(push)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#pragma warning(pop)
#pragma GCC diagnostic pop
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/Host.h>

#include "Runtime.h"
#include "Compiler.h"
#include "Cache.h"
#include "BuildInfo.gen.h"

#include <iostream>

namespace dev
{
namespace eth
{
namespace jit
{

namespace
{
typedef ReturnCode(*EntryFuncPtr)(Runtime*);

ReturnCode runEntryFunc(EntryFuncPtr _mainFunc, Runtime* _runtime)
{
	// That function uses long jumps to handle "execeptions".
	// Do not create any non-POD objects here

	ReturnCode returnCode{};
	auto sj = setjmp(_runtime->getJmpBuf());
	if (sj == 0)
		returnCode = _mainFunc(_runtime);
	else
		returnCode = static_cast<ReturnCode>(sj);

	return returnCode;
}

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

bool getEnvOption(char const* _name, bool _default)
{
	auto var = std::getenv(_name);
	if (!var)
		return _default;
	return std::strtol(var, nullptr, 10) != 0;
}

bool showInfo()
{
	auto show = getEnvOption("EVMJIT_INFO", false);
	if (show)
	{
		std::cout << "The Ethereum EVM JIT " EVMJIT_VERSION_FULL " LLVM " LLVM_VERSION << std::endl;
	}
	return show;
}

class StatsCollector
{
public:
	std::vector<std::unique_ptr<ExecStats>> stats;

	~StatsCollector()
	{
		if (stats.empty())
			return;

		using d = decltype(ExecStats{}.execTime);
		d total = d::zero();
		d max = d::zero();
		d min = d::max();

		for (auto&& s : stats)
		{
			auto t = s->execTime;
			total += t;
			if (t < min)
				min = t;
			if (t > max)
				max = t;
		}

		using u = std::chrono::microseconds;
		auto nTotal = std::chrono::duration_cast<u>(total).count();
		auto nAverage = std::chrono::duration_cast<u>(total / stats.size()).count();
		auto nMax = std::chrono::duration_cast<u>(max).count();
		auto nMin = std::chrono::duration_cast<u>(min).count();

		std::cout << "Total  exec time: " << nTotal << " us" << std::endl
				  << "Averge exec time: " << nAverage << " us" << std::endl
				  << "Min    exec time: " << nMin << " us" << std::endl
				  << "Max    exec time: " << nMax << " us" << std::endl;
	}
};

}

void ExecutionEngine::collectStats()
{
	if (!m_stats)
		m_stats.reset(new ExecStats);
}

ReturnCode ExecutionEngine::run(RuntimeData* _data, Env* _env)
{
	static std::unique_ptr<llvm::ExecutionEngine> ee;  // TODO: Use Managed Objects from LLVM?
	static auto debugDumpModule = getEnvOption("EVMJIT_DUMP", false);
	static auto objectCacheEnabled = getEnvOption("EVMJIT_CACHE", true);
	static auto statsCollectingEnabled = getEnvOption("EVMJIT_STATS", false);
	static auto infoShown = showInfo();
	(void) infoShown;

	static StatsCollector statsCollector;

	if (statsCollectingEnabled)
		collectStats();

	auto codeBegin = _data->code;
	auto codeEnd = codeBegin + _data->codeSize;
	assert(codeBegin || !codeEnd); //TODO: Is it good idea to execute empty code?
	auto mainFuncName = codeHash(_data->codeHash);
	EntryFuncPtr entryFuncPtr{};
	Runtime runtime(_data, _env);	// TODO: I don't know why but it must be created before getFunctionAddress() calls

	if (ee && (entryFuncPtr = (EntryFuncPtr)ee->getFunctionAddress(mainFuncName)))
	{
	}
	else
	{
		auto objectCache = objectCacheEnabled ? Cache::getObjectCache() : nullptr;
		std::unique_ptr<llvm::Module> module;
		if (objectCache)
			module = Cache::getObject(mainFuncName);
		if (!module)
			module = Compiler({}).compile(codeBegin, codeEnd, mainFuncName);
		if (debugDumpModule)
			module->dump();
		if (!ee)
		{
			llvm::InitializeNativeTarget();
			llvm::InitializeNativeTargetAsmPrinter();

			llvm::EngineBuilder builder(module.get());
			builder.setEngineKind(llvm::EngineKind::JIT);
			builder.setUseMCJIT(true);
			std::unique_ptr<llvm::SectionMemoryManager> memoryManager(new llvm::SectionMemoryManager);
			builder.setMCJITMemoryManager(memoryManager.get());
			builder.setOptLevel(llvm::CodeGenOpt::None);
			builder.setVerifyModules(true);

			auto triple = llvm::Triple(llvm::sys::getProcessTriple());
			if (triple.getOS() == llvm::Triple::OSType::Win32)
				triple.setObjectFormat(llvm::Triple::ObjectFormatType::ELF);  // MCJIT does not support COFF format
			module->setTargetTriple(triple.str());

			ee.reset(builder.create());
			if (!ee)
				return ReturnCode::LLVMConfigError;

			module.release();         // Successfully created llvm::ExecutionEngine takes ownership of the module
			memoryManager.release();  // and memory manager

			if (objectCache)
				ee->setObjectCache(objectCache);
			entryFuncPtr = (EntryFuncPtr)ee->getFunctionAddress(mainFuncName);
		}
		else
		{
			if (!entryFuncPtr)
			{
				ee->addModule(module.get());
				module.release();
				entryFuncPtr = (EntryFuncPtr)ee->getFunctionAddress(mainFuncName);
			}
		}
	}
	assert(entryFuncPtr);

	if (m_stats)
		m_stats->execStarted();

	auto returnCode = runEntryFunc(entryFuncPtr, &runtime);

	if (m_stats)
		m_stats->execEnded();

	if (returnCode == ReturnCode::Return)
	{
		returnData = runtime.getReturnData();     // Save reference to return data
		std::swap(m_memory, runtime.getMemory()); // Take ownership of memory
	}

	if (statsCollectingEnabled)
		statsCollector.stats.push_back(std::move(m_stats));

	return returnCode;
}

}
}
}
