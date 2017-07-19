#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS 1

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#endif

#include "CL/cl.hpp"

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <time.h>
#include <functional>
#include <libethash/ethash.h>

namespace dev { namespace eth { class EthashCLHook; }}

class ethash_cl_miner
{
private:
	enum { c_maxSearchResults = 1 };

public:
	ethash_cl_miner() = default;

	bool init(
		ethash_light_t _light,
		uint8_t const* _lightData,
		uint64_t _lightSize,
		unsigned _platformId,
		unsigned _deviceId,
		unsigned _workgroupSize,
		unsigned initialGlobalWorkSize
		);
	void search(uint8_t const* _header, uint64_t _target, dev::eth::EthashCLHook& _hook, uint64_t _startN);

private:
	cl::Context m_context;
	cl::CommandQueue m_queue;
	cl::Kernel m_searchKernel;
	cl::Kernel m_dagKernel;
	cl::Buffer m_dag;
	cl::Buffer m_light;
	cl::Buffer m_header;
	cl::Buffer m_searchBuffer;
	unsigned m_globalWorkSize = 0;
	unsigned m_workgroupSize = 0;
};
