#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct evmjit_result
{
	int32_t  returnCode;
	uint64_t returnDataSize;
	void*    returnData;

} evmjit_result;

evmjit_result evmjit_run(void* _data, void* _env);

#ifdef __cplusplus
}
#endif
