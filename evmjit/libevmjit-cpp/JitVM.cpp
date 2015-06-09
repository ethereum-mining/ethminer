
#pragma GCC diagnostic ignored "-Wconversion"

#include "JitVM.h"

#include <libdevcore/Log.h>
#include <libdevcore/SHA3.h>
#include <libevm/VM.h>
#include <libevm/VMFactory.h>
#include <evmjit/libevmjit/ExecutionEngine.h>

#include "Utils.h"

namespace dev
{
namespace eth
{

extern "C" void env_sload(); // fake declaration for linker symbol stripping workaround, see a call below

bytesConstRef JitVM::execImpl(u256& io_gas, ExtVMFace& _ext, OnOpFunc const& _onOp)
{
	using namespace jit;

	auto rejected = false;
	// TODO: Rejecting transactions with gas limit > 2^63 can be used by attacker to take JIT out of scope
	rejected |= io_gas > std::numeric_limits<decltype(m_data.gas)>::max(); // Do not accept requests with gas > 2^63 (int64 max)
	rejected |= _ext.gasPrice > std::numeric_limits<decltype(m_data.gasPrice)>::max();
	rejected |= _ext.currentBlock.number > std::numeric_limits<decltype(m_data.number)>::max();
	rejected |= _ext.currentBlock.timestamp > std::numeric_limits<decltype(m_data.timestamp)>::max();

	if (rejected)
	{
		cwarn << "Execution rejected by EVM JIT (gas limit: " << io_gas << "), executing with interpreter";
		m_fallbackVM = VMFactory::create(VMKind::Interpreter);
		return m_fallbackVM->execImpl(io_gas, _ext, _onOp);
	}

	m_data.gas 			= static_cast<decltype(m_data.gas)>(io_gas);
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
	m_data.codeHash		= eth2llvm(sha3(_ext.code));

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
	case ReturnCode::StackUnderflow:
		BOOST_THROW_EXCEPTION(StackUnderflow());
	case ReturnCode::BadInstruction:
		BOOST_THROW_EXCEPTION(BadInstruction());
	case ReturnCode::LinkerWorkaround:	// never happens
		env_sload();					// but forces linker to include env_* JIT callback functions
		break;
	default:
		break;
	}

	io_gas = m_data.gas;
	return {std::get<0>(m_engine.returnData), std::get<1>(m_engine.returnData)};
}

}
}
