
#include "JitVM.h"
#include <libevm/VM.h>
#include <evmjit/libevmjit/ExecutionEngine.h>
#include "Utils.h"

namespace dev
{
namespace eth
{

extern "C" void env_sload(); // fake declaration for linker symbol stripping workaround, see a call below

bytesConstRef JitVM::go(ExtVMFace& _ext, OnOpFunc const&, uint64_t)
{
	using namespace jit;

	if (m_gas > std::numeric_limits<decltype(m_data.gas)>::max())
		BOOST_THROW_EXCEPTION(OutOfGas()); // Do not accept requests with gas > 2^63 (int64 max) // TODO: Return "not accepted" exception to allow interpreted handle that

	if (_ext.gasPrice > std::numeric_limits<decltype(m_data.gasPrice)>::max())
		BOOST_THROW_EXCEPTION(OutOfGas());

	if (_ext.currentBlock.number > std::numeric_limits<decltype(m_data.number)>::max())
		BOOST_THROW_EXCEPTION(OutOfGas());

	if (_ext.currentBlock.timestamp > std::numeric_limits<decltype(m_data.timestamp)>::max())
		BOOST_THROW_EXCEPTION(OutOfGas());


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
	auto exitCode = m_engine.run(_ext.code, &m_data, env);
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
	case ReturnCode::LinkerWorkaround:	// never happens
		env_sload();					// but forces linker to include env_* JIT callback functions
		break;
	default:
		break;
	}

	m_gas = m_data.gas; // TODO: Remove m_gas field
	return {std::get<0>(m_engine.returnData), std::get<1>(m_engine.returnData)};
}

}
}
