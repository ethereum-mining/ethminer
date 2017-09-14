#pragma once

//#include <cuda_runtime.h>

#include <time.h>
#include <functional>
#include <libethash/ethash.h>
#include <libethcore/Miner.h>
#include <libhwmon/wrapnvml.h>
#include "ethash_cuda_miner_kernel.h"

class ethash_cuda_miner
{
public:
	struct search_hook
	{
		virtual ~search_hook(); // always a virtual destructor for a class with virtuals.

		// reports progress, return true to abort
		virtual bool found(uint64_t const* nonces, uint32_t count) = 0;
		virtual bool searched(uint64_t start_nonce, uint32_t count) = 0;
	};

public:
	ethash_cuda_miner();

	static std::string platform_info(unsigned _deviceId = 0);
	static int getNumDevices();
	static void listDevices();
	static bool configureGPU(
		int *	 _devices,
		unsigned _blockSize,
		unsigned _gridSize,
		unsigned _numStreams,
		unsigned _scheduleFlag,
		uint64_t _currentBlock
		);
        static void setParallelHash(unsigned _parallelHash);

	bool init(ethash_light_t _light, uint8_t const* _lightData, uint64_t _lightSize, unsigned _deviceId, bool _cpyToHost, volatile void** hostDAG);

	void finish();
	void search(uint8_t const* header, uint64_t target, search_hook& hook, bool _ethStratum, uint64_t _startN);
	dev::eth::HwMonitor hwmon();

	/* -- default values -- */
	/// Default value of the block size. Also known as workgroup size.
	static unsigned const c_defaultBlockSize;
	/// Default value of the grid size
	static unsigned const c_defaultGridSize;
	// default number of CUDA streams
	static unsigned const c_defaultNumStreams;

private:
	hash32_t m_current_header;
	uint64_t m_current_target;
	uint64_t m_current_nonce;
	uint64_t m_starting_nonce;
	uint64_t m_current_index;
	uint32_t m_sharedBytes;
	int m_device_num;

	volatile uint32_t ** m_search_buf;
	cudaStream_t  * m_streams;

	/// The local work size for the search
	static unsigned s_blockSize;
	/// The initial global work size for the searches
	static unsigned s_gridSize;
	/// The number of CUDA streams
	static unsigned s_numStreams;
	/// CUDA schedule flag
	static unsigned s_scheduleFlag;

	static unsigned m_parallelHash;

	wrap_nvml_handle *nvmlh = NULL;
};
