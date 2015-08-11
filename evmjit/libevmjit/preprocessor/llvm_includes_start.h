#if defined(_MSC_VER)
	#pragma warning(push)
	#pragma warning(disable: 4267 4244 4800 4624)
#elif defined(__clang__)
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wunused-parameter"
	#pragma clang diagnostic ignored "-Wconversion"
#else
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wunused-parameter"
	#pragma GCC diagnostic ignored "-Wconversion"
#endif
