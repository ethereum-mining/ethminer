
#include "stdint.h"

#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#define _ALLOW_KEYWORD_MACROS
#define noexcept throw()
#else
#define EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct evmjit_i256
{
	uint64_t words[4];
} evmjit_i256;

typedef struct evmjit_runtime_data
{
	int64_t 	gas;
	int64_t 	gasPrice;
	char const* callData;
	uint64_t 	callDataSize;
	evmjit_i256 address;
	evmjit_i256 caller;
	evmjit_i256 origin;
	evmjit_i256 callValue;
	evmjit_i256 coinBase;
	evmjit_i256 difficulty;
	evmjit_i256 gasLimit;
	uint64_t 	number;
	int64_t 	timestamp;
	char const* code;
	uint64_t 	codeSize;
	evmjit_i256	codeHash;
} evmjit_runtime_data;

typedef enum evmjit_return_code
{
	// Success codes
	Stop    = 0,
	Return  = 1,
	Suicide = 2,

	// Standard error codes
	OutOfGas           = -1,
	StackUnderflow     = -2,
	BadJumpDestination = -3,
	BadInstruction     = -4,
	Rejected           = -5, ///< Input data (code, gas, block info, etc.) does not meet JIT requirement and execution request has been rejected

	// Internal error codes
	LLVMError           = -101,
	UnexpectedException = -111
} evmjit_return_code;

typedef struct evmjit_context evmjit_context;

EXPORT evmjit_context* evmjit_create(evmjit_runtime_data* _data, void* _env);

EXPORT evmjit_return_code evmjit_exec(evmjit_context* _context);

EXPORT void evmjit_destroy(evmjit_context* _context);


inline char const* evmjit_get_output(evmjit_runtime_data* _data) { return _data->callData; }
inline uint64_t evmjit_get_output_size(evmjit_runtime_data* _data) { return _data->callDataSize; }

#ifdef __cplusplus
}
#endif
