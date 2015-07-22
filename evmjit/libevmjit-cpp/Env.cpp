
#pragma GCC diagnostic ignored "-Wconversion"
#include <libdevcore/SHA3.h>
#include <libevmcore/Params.h>
#include <libevm/ExtVMFace.h>

#include "Utils.h"

extern "C"
{
	#ifdef _MSC_VER
		#define EXPORT __declspec(dllexport)
	#else
		#define EXPORT
	#endif

	using namespace dev;
	using namespace dev::eth;
	using evmjit::i256;

	EXPORT void env_sload(ExtVMFace* _env, i256* _index, i256* o_value)
	{
		auto index = jit2eth(*_index);
		auto value = _env->store(index); // Interface uses native endianness
		*o_value = eth2jit(value);
	}

	EXPORT void env_sstore(ExtVMFace* _env, i256* _index, i256* _value)
	{
		auto index = jit2eth(*_index);
		auto value = jit2eth(*_value);

		if (value == 0 && _env->store(index) != 0)	// If delete
			_env->sub.refunds += c_sstoreRefundGas;	// Increase refund counter

		_env->setStore(index, value);	// Interface uses native endianness
	}

	EXPORT void env_balance(ExtVMFace* _env, h256* _address, i256* o_value)
	{
		auto u = _env->balance(right160(*_address));
		*o_value = eth2jit(u);
	}

	EXPORT void env_blockhash(ExtVMFace* _env, i256* _number, h256* o_hash)
	{
		*o_hash = _env->blockHash(jit2eth(*_number));
	}

	EXPORT void env_create(ExtVMFace* _env, int64_t* io_gas, i256* _endowment, byte* _initBeg, uint64_t _initSize, h256* o_address)
	{
		auto endowment = jit2eth(*_endowment);
		if (_env->balance(_env->myAddress) >= endowment && _env->depth < 1024)
		{
			u256 gas = *io_gas;
			h256 address(_env->create(endowment, gas, {_initBeg, (size_t)_initSize}, {}), h256::AlignRight);
			*io_gas = static_cast<int64_t>(gas);
			*o_address = address;
		}
		else
			*o_address = {};
	}

	EXPORT bool env_call(ExtVMFace* _env, int64_t* io_gas, int64_t _callGas, h256* _receiveAddress, i256* _value, byte* _inBeg, uint64_t _inSize, byte* _outBeg, uint64_t _outSize, h256* _codeAddress)
	{
		CallParameters params;
		params.value = jit2eth(*_value);
		params.senderAddress = _env->myAddress;
		params.receiveAddress = right160(*_receiveAddress);
		params.codeAddress = right160(*_codeAddress);
		params.data = {_inBeg, (size_t)_inSize};
		params.out = {_outBeg, (size_t)_outSize};
		params.onOp = {};
		const auto isCall = params.receiveAddress == params.codeAddress; // OPT: The same address pointer can be used if not CODECALL

		*io_gas -= _callGas;
		if (*io_gas < 0)
			return false;

		if (isCall && !_env->exists(params.receiveAddress))
			*io_gas -= static_cast<int64_t>(c_callNewAccountGas); // no underflow, *io_gas non-negative before

		if (params.value > 0) // value transfer
		{
			/*static*/ assert(c_callValueTransferGas > c_callStipend && "Overflow possible");
			*io_gas -= static_cast<int64_t>(c_callValueTransferGas); // no underflow
			_callGas += static_cast<int64_t>(c_callStipend); // overflow possibility, but in the same time *io_gas < 0
		}

		if (*io_gas < 0)
			return false;

		auto ret = false;
		params.gas = u256{_callGas};
		if (_env->balance(_env->myAddress) >= params.value && _env->depth < 1024)
			ret = _env->call(params);

		*io_gas += static_cast<int64_t>(params.gas); // it is never more than initial _callGas
		return ret;
	}

	EXPORT void env_sha3(byte* _begin, uint64_t _size, h256* o_hash)
	{
		auto hash = sha3({_begin, (size_t)_size});
		*o_hash = hash;
	}

	EXPORT byte const* env_extcode(ExtVMFace* _env, h256* _addr256, uint64_t* o_size)
	{
		auto addr = right160(*_addr256);
		auto& code = _env->codeAt(addr);
		*o_size = code.size();
		return code.data();
	}

	EXPORT void env_log(ExtVMFace* _env, byte* _beg, uint64_t _size, h256* _topic1, h256* _topic2, h256* _topic3, h256* _topic4)
	{
		dev::h256s topics;

		if (_topic1)
			topics.push_back(*_topic1);

		if (_topic2)
			topics.push_back(*_topic2);

		if (_topic3)
			topics.push_back(*_topic3);

		if (_topic4)
			topics.push_back(*_topic4);

		_env->log(std::move(topics), {_beg, (size_t)_size});
	}
}

