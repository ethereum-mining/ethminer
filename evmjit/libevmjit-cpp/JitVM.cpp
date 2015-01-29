
#include "JitVM.h"
#include <libevm/VM.h>
#include <libevm/VMFactory.h>
#include <evmjit/libevmjit/ExecutionEngine.h>
#include "Utils.h"

namespace dev
{
namespace eth
{

bytesConstRef JitVM::go(ExtVMFace& _ext, OnOpFunc const& _onOp, uint64_t _step)
{
	using namespace jit;

	auto rejected = false;
	rejected |= m_gas > std::numeric_limits<decltype(m_data.gas)>::max(); // Do not accept requests with gas > 2^63 (int64 max)
	rejected |= _ext.gasPrice > std::numeric_limits<decltype(m_data.gasPrice)>::max();
	rejected |= _ext.currentBlock.number > std::numeric_limits<decltype(m_data.number)>::max();
	rejected |= _ext.currentBlock.timestamp > std::numeric_limits<decltype(m_data.timestamp)>::max();

	if (rejected)
	{
		UNTESTED;
		std::cerr << "Rejected\n";
		VMFactory::setKind(VMKind::Interpreter);
		m_fallbackVM = VMFactory::create(m_gas);
		VMFactory::setKind(VMKind::JIT);
		return m_fallbackVM->go(_ext, _onOp, _step);
	}

	m_data.gas 			= static_cast<decltype(m_data.gas)>(m_gas);
	m_data.gasPrice		= static_cast<decltype(m_data.gasPrice)>(_ext.gasPrice);
	m_data.callData 	= _ext.data.data();
	m_data.callDataSize = _ext.data.size();
	m_data.address      = eth2llvm(fromAddress(_ext.myAddress));
	m_data.caller       = eth2llvm(fromAddress(_ext.caller));
	m_data.origin       = eth2llvm(fromAddress(_ext.origin));
	m_data.callValue    = eth2llvm(_ext.value);
	m_data.coinBase     = eth2llvm(fromAddress(_ext.currentBlock.coinbaseAddress));
	m_data.difficulty   = eth2llvm(_ext.currentBlock.difficulty);
	m_data.gasLimit     = eth2llvm(_ext.currentBlock.gasLimit);
	m_data.number 		= static_cast<decltype(m_data.number)>(_ext.currentBlock.number);
	m_data.timestamp 	= static_cast<decltype(m_data.timestamp)>(_ext.currentBlock.timestamp);
	m_data.code     	= _ext.code.data();
	m_data.codeSize 	= _ext.code.size();

	auto env = reinterpret_cast<Env*>(&_ext);
	auto exitCode = m_engine.run(&m_data, env);
	switch (exitCode)
	{
	case ReturnCode::Suicide:
		_ext.suicide(right160(llvm2eth(m_data.address)));
		break;

	case ReturnCode::BadJumpDestination:
		BOOST_THROW_EXCEPTION(BadJumpDestination());
	case ReturnCode::OutOfGas:
		BOOST_THROW_EXCEPTION(OutOfGas());
	case ReturnCode::StackTooSmall:
		BOOST_THROW_EXCEPTION(StackTooSmall());
	case ReturnCode::BadInstruction:
		BOOST_THROW_EXCEPTION(BadInstruction());
	default:
		break;
	}

	m_gas = m_data.gas; // TODO: Remove m_gas field
	return {std::get<0>(m_engine.returnData), std::get<1>(m_engine.returnData)};
}

}
}

namespace
{
	// MSVS linker ignores export symbols in Env.cpp if nothing points at least one of them
	extern "C" void env_sload();
	void linkerWorkaround() 
	{ 
		env_sload();
		(void)&linkerWorkaround; // suppress unused function warning from GCC
	}
}
