#pragma once

//#include <cuda_runtime.h>

#include <time.h>
#include <functional>
#include <libethash/ethash.h>
#include "ethash_cu_miner_kernel.h"

class ethash_cu_miner
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
	ethash_cu_miner();

	static std::string platform_info(unsigned _deviceId = 0);
	static unsigned getNumDevices();
	static void listDevices();
	static bool configureGPU(
		unsigned _blockSize,
		unsigned _gridSize,
		unsigned _numStreams,
		unsigned _extraGPUMemory,
		bool	 _highcpu,
		uint64_t _currentBlock
		);
	bool init(
		uint8_t const* _dag,
		uint64_t _dagSize,
		unsigned _deviceId = 0
		);
	void finish();
	void search(uint8_t const* header, uint64_t target, search_hook& hook);

	/* -- default values -- */
	/// Default value of the block size. Also known as workgroup size.
	static unsigned const c_defaultBlockSize;
	/// Default value of the grid size
	static unsigned const c_defaultGridSize;
	// default number of CUDA streams
	static unsigned const c_defaultNumStreams;

private:
	enum { c_max_search_results = 63, c_hash_batch_size = 1024 };

	hash128_t * m_dag_ptr;
	hash32_t * m_header;

	void ** m_hash_buf;
	uint32_t ** m_search_buf;
	cudaStream_t  * m_streams;

	/// The local work size for the search
	static unsigned s_blockSize;
	/// The initial global work size for the searches
	static unsigned s_gridSize;
	/// The number of CUDA streams
	static unsigned s_numStreams;
	/// Whether or not to let the CPU wait
	static bool s_highCPU;

	/// GPU memory required for other things, like window rendering e.t.c.
	/// User can set it via the --cl-extragpu-mem argument.
	static unsigned s_extraRequiredGPUMem;
};