/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/

// SmartVM is only available if EVM JIT is enabled
#if ETH_EVMJIT

#include "SmartVM.h"
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <libdevcore/Log.h>
#include <libdevcore/SHA3.h>
#include <libdevcore/Guards.h>
#include <evmjit/JIT.h>
#include <evmjit/libevmjit-cpp/Utils.h>
#include "VMFactory.h"

namespace dev
{

template<typename _T, typename _QueueT = std::queue<_T>>
class concurrent_queue
{
public:
	template<typename _U>
	void push(_U&& _elem)
	{
		{
			std::lock_guard<decltype(x_mutex)> guard{x_mutex};
			m_queue.push(std::forward<_U>(_elem));
		}
		m_cv.notify_one();
	}

	_T pop()
	{
		std::unique_lock<std::mutex> lock{x_mutex};
		m_cv.wait(lock, [this]{ return !m_queue.empty(); });
		auto item = std::move(m_queue.front());
		m_queue.pop();
		return item;
	}

private:
	_QueueT m_queue;
	std::mutex x_mutex;
	std::condition_variable m_cv;
};

namespace eth
{
namespace
{
	struct JitInfo: LogChannel { static const char* name() { return "JIT"; }; static const int verbosity = 11; };

	using HitMap = std::unordered_map<h256, uint64_t>;

	HitMap& getHitMap()
	{
		static HitMap s_hitMap;
		return s_hitMap;
	}

	struct JitTask
	{
		bytes code;
		h256 codeHash;

		static JitTask createStopSentinel() { return {}; }

		bool isStopSentinel()
		{
			assert((!code.empty() || !codeHash) && "'empty code => empty hash' invariand failed");
			return code.empty();
		}
	};

	class JitWorker
	{
		std::thread m_worker;
		concurrent_queue<JitTask> m_queue;

		void work()
		{
			clog(JitInfo) << "JIT worker started.";
			JitTask task;
			while (!(task = m_queue.pop()).isStopSentinel())
			{
				clog(JitInfo) << "Compilation... " << task.codeHash;
				evmjit::JIT::compile(task.code.data(), task.code.size(), eth2jit(task.codeHash));
				clog(JitInfo) << "   ...finished " << task.codeHash;
			}
			clog(JitInfo) << "JIT worker finished.";
		}

	public:
		JitWorker() noexcept: m_worker([this]{ work(); })
		{}

		~JitWorker()
		{
			push(JitTask::createStopSentinel());
			m_worker.join();
		}

		void push(JitTask&& _task) { m_queue.push(std::move(_task)); }
	};
}

bytesConstRef SmartVM::execImpl(u256& io_gas, ExtVMFace& _ext, OnOpFunc const& _onOp)
{
	auto codeHash = _ext.codeHash;
	auto vmKind = VMKind::Interpreter; // default VM

	// Jitted EVM code already in memory?
	if (evmjit::JIT::isCodeReady(eth2jit(codeHash)))
	{
		clog(JitInfo) << "JIT:           " << codeHash;
		vmKind = VMKind::JIT;
	}
	else
	{
		static JitWorker s_worker;

		// Check EVM code hit count
		static const uint64_t c_hitTreshold = 2;
		auto& hits = getHitMap()[codeHash];
		++hits;
		if (hits == c_hitTreshold)
		{
			clog(JitInfo) << "Schedule:      " << codeHash;
			s_worker.push({_ext.code, codeHash});
		}
		clog(JitInfo) << "Interpreter:   " << codeHash;
	}

	// TODO: Selected VM must be kept only because it returns reference to its internal memory.
	//       VM implementations should be stateless, without escaping memory reference.
	m_selectedVM = VMFactory::create(vmKind);
	return m_selectedVM->execImpl(io_gas, _ext, _onOp);
}

}
}

#endif
