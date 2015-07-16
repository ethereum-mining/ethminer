
#pragma GCC diagnostic ignored "-Wconversion"

#include "JitVM.h"

#include <libdevcore/Log.h>
#include <libdevcore/SHA3.h>
#include <libevm/VM.h>
#include <libevm/VMFactory.h>

#include "Utils.h"

namespace dev
{
namespace eth
{

extern "C" void env_sload(); // fake declaration for linker symbol stripping workaround, see a call below

bytesConstRef JitVM::execImpl(u256& io_gas, ExtVMFace& _ext, OnOpFunc const& _onOp)
{
	auto rejected = false;
	// TODO: Rejecting transactions with gas limit > 2^63 can be used by attacker to take JIT out of scope
	rejected |= io_gas > std::numeric_limits<decltype(m_data.gas)>::max(); // Do not accept requests with gas > 2^63 (int64 max)
	rejected |= _ext.gasPrice > std::numeric_limits<decltype(m_data.gasPrice)>::max();
	rejected |= _ext.currentBlock.number() > std::numeric_limits<decltype(m_data.number)>::max();
	rejected |= _ext.currentBlock.timestamp() > std::numeric_limits<decltype(m_data.timestamp)>::max();

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
	m_data.address      = eth2jit(fromAddress(_ext.myAddress));
	m_data.caller       = eth2jit(fromAddress(_ext.caller));
	m_data.origin       = eth2jit(fromAddress(_ext.origin));
	m_data.callValue    = eth2jit(_ext.value);
	m_data.coinBase     = eth2jit(fromAddress(_ext.currentBlock.coinbaseAddress()));
	m_data.difficulty   = eth2jit(_ext.currentBlock.difficulty());
	m_data.gasLimit     = eth2jit(_ext.currentBlock.gasLimit());
	m_data.number 		= static_cast<decltype(m_data.number)>(_ext.currentBlock.number());
	m_data.timestamp 	= static_cast<decltype(m_data.timestamp)>(_ext.currentBlock.timestamp());
	m_data.code     	= _ext.code.data();
	m_data.codeSize 	= _ext.code.size();
	m_data.codeHash		= eth2jit(_ext.codeHash);

	// Pass pointer to ExtVMFace casted to evmjit::Env* opaque type.
	// JIT will do nothing with the pointer, just pass it to Env callback functions implemented in Env.cpp.
	m_context.init(m_data, reinterpret_cast<evmjit::Env*>(&_ext));
	auto exitCode = evmjit::JIT::exec(m_context);
	switch (exitCode)
	{
	case evmjit::ReturnCode::Suicide:
		_ext.suicide(right160(jit2eth(m_data.address)));
		break;

	case evmjit::ReturnCode::BadJumpDestination:
		BOOST_THROW_EXCEPTION(BadJumpDestination());
	case evmjit::ReturnCode::OutOfGas:
		BOOST_THROW_EXCEPTION(OutOfGas());
	case evmjit::ReturnCode::StackUnderflow: // FIXME: Remove support for detail errors
		BOOST_THROW_EXCEPTION(StackUnderflow());
	case evmjit::ReturnCode::BadInstruction:
		BOOST_THROW_EXCEPTION(BadInstruction());
	case evmjit::ReturnCode::LinkerWorkaround:	// never happens
		env_sload();					// but forces linker to include env_* JIT callback functions
		break;
	default:
		break;
	}

	io_gas = m_data.gas;
	return {std::get<0>(m_context.returnData), std::get<1>(m_context.returnData)};
}

}
}
