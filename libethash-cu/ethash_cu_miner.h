#pragma once

#include <cuda_runtime.h>

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

	bool init(uint8_t const* _dag, uint64_t _dagSize, unsigned num_buffers = 2, unsigned search_batch_size = 262144, unsigned workgroup_size = 64, unsigned _deviceId = 0, bool highcpu = false);
	static std::string platform_info(unsigned _deviceId = 0);
	static int get_num_devices();


	void finish();
	void hash(uint8_t* ret, uint8_t const* header, uint64_t nonce, unsigned count);
	void search(uint8_t const* header, uint64_t target, search_hook& hook);

	/* -- default values -- */
	/// Default value of the local work size. Also known as workgroup size.
	static unsigned const c_defaultLocalWorkSize;
	/// Default value of the global work size as a multiplier of the local work size
	static unsigned const c_defaultGlobalWorkSizeMultiplier;

private:
	enum { c_max_search_results = 63, c_hash_batch_size = 1024 };
	
	bool	 m_highcpu;
	unsigned m_num_buffers;
	unsigned m_search_batch_size;
	unsigned m_workgroup_size;

	hash128_t * m_dag_ptr;
	hash32_t * m_header;

	void ** m_hash_buf;
	uint32_t ** m_search_buf;
	cudaStream_t  * m_streams;

	
};