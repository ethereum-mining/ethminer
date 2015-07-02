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
#include <evmjit/JIT.h>
#include <evmjit/libevmjit-cpp/Utils.h>
#include "VMFactory.h"

namespace dev
{
namespace eth
{
namespace
{
	struct JitInfo: LogChannel { static const char* name() { return "JIT"; }; static const int verbosity = 0; };

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
	};

	class JitWorker
	{
		bool m_finished = false;
		std::mutex x_mutex;
		std::condition_variable m_cv;
		std::thread m_worker;
		std::queue<JitTask> m_queue;

		bool pop(JitTask& o_task)
		{
			std::unique_lock<std::mutex> lock{x_mutex};
			m_cv.wait(lock, [this]{ return m_finished || !m_queue.empty(); });
			if (m_finished)
				return false;

			assert(!m_queue.empty());
			o_task = std::move(m_queue.front());
			m_queue.pop();
			return true;
		}

		void work()
		{
			clog(JitInfo) << "JIT worker started.";
			JitTask task;
			while (pop(task))
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
			{
				std::lock_guard<std::mutex> lock{x_mutex};
				m_finished = true;
			}
			m_cv.notify_one();
			m_worker.join();
		}

		void push(JitTask&& _task)
		{
			{
		        std::lock_guard<std::mutex> lock(x_mutex);
		        m_queue.push(std::move(_task));
		    }
			m_cv.notify_one();
		}
	};
}

bytesConstRef SmartVM::execImpl(u256& io_gas, ExtVMFace& _ext, OnOpFunc const& _onOp)
{
	auto codeHash = _ext.codeHash;
	auto vmKind = VMKind::Interpreter; // default VM

	// Jitted EVM code already in memory?
	if (evmjit::JIT::isCodeReady(eth2jit(codeHash))) // FIXME: JIT::isCodeReady is not thread-safe
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
