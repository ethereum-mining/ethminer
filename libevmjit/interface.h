
#ifdef __cplusplus
extern "C" {
#endif

void* evmjit_create();
int   evmjit_run(void* _jit, void* _data, void* _env);
void  evmjit_destroy(void* _jit);


#ifdef __cplusplus
}
#endif
