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

class ethash_cl_miner
{
private:
	enum { c_maxSearchResults = 1 };

public:
	struct search_hook
	{
		virtual ~search_hook(); // always a virtual destructor for a class with virtuals.

		// reports progress, return true to abort
		virtual bool found(uint64_t const* nonces, uint32_t count) = 0;
		virtual bool searched(uint64_t start_nonce, uint32_t count) = 0;
	};

	ethash_cl_miner() = default;

	static bool searchForAllDevices(unsigned _platformId, std::function<bool(cl::Device const&)> _callback);
	static void doForAllDevices(unsigned _platformId, std::function<void(cl::Device const&)> _callback);
	static void doForAllDevices(std::function<void(cl::Device const&)> _callback);
	static unsigned getNumDevices(unsigned _platformId = 0);
	static std::string platform_info(unsigned _platformId = 0, unsigned _deviceId = 0);
	static void listDevices();
	static bool configureGPU(
		unsigned _platformId,
		unsigned _localWorkSize,
		unsigned _globalWorkSize,
		unsigned _extraGPUMemory,
		uint64_t _currentBlock
	);

	bool init(
		ethash_light_t _light,
		uint8_t const* _lightData,
		uint64_t _lightSize,
		unsigned _platformId,
		unsigned _deviceId
		);
	void search(uint8_t const* _header, uint64_t _target, search_hook& _hook, uint64_t _startN);

	/* -- default values -- */
	/// Default value of the local work size. Also known as workgroup size.
	static unsigned const c_defaultLocalWorkSize;
	/// Default value of the global work size as a multiplier of the local work size
	static unsigned const c_defaultGlobalWorkSizeMultiplier;

private:

	static std::vector<cl::Device> getDevices(std::vector<cl::Platform> const& _platforms, unsigned _platformId);
	static std::vector<cl::Platform> getPlatforms();

	cl::Context m_context;
	cl::CommandQueue m_queue;
	cl::Kernel m_searchKernel;
	cl::Kernel m_dagKernel;
	cl::Buffer m_dag;
	cl::Buffer m_light;
	cl::Buffer m_header;
	cl::Buffer m_searchBuffer;
	unsigned m_globalWorkSize;

	/// The local work size for the search
	static unsigned s_workgroupSize;
	/// The initial global work size for the searches
	static unsigned s_initialGlobalWorkSize;
	/// GPU memory required for other things, like window rendering e.t.c.
	/// User can set it via the --cl-extragpu-mem argument.
	static unsigned s_extraRequiredGPUMem;
};
