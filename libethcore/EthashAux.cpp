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
/** @file EthashAux.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "EthashAux.h"

#include <boost/detail/endian.hpp>
#include <boost/filesystem.hpp>
#include <chrono>
#include <array>
#include <thread>
#include <libdevcore/Common.h>
#include <libdevcore/Guards.h>
#include <libdevcore/Log.h>
#include <libdevcore/SHA3.h>
#include <libdevcore/FileSystem.h>
#include <libethash/internal.h>
#include <libethash/io.h>
#include "BlockInfo.h"
#include "Exceptions.h"
using namespace std;
using namespace chrono;
using namespace dev;
using namespace eth;
using namespace boost::filesystem;

const char* DAGChannel::name() { return EthGreen "DAG"; }

EthashAux* dev::eth::EthashAux::s_this = nullptr;
char  dev::eth::EthashAux::s_dagDirName[256] = "";
dev::eth::DAGEraseMode dev::eth::EthashAux::s_dagEraseMode = DAGEraseMode::None;

const unsigned EthashProofOfWork::defaultLocalWorkSize = 64;
const unsigned EthashProofOfWork::defaultGlobalWorkSizeMultiplier = 4096; // * CL_DEFAULT_LOCAL_WORK_SIZE
const unsigned EthashProofOfWork::defaultMSPerBatch = 0;
const EthashProofOfWork::WorkPackage EthashProofOfWork::NullWorkPackage = EthashProofOfWork::WorkPackage();

//unsigned EthashProofOfWork::s_dagLoadMode = 0;

EthashAux::~EthashAux()
{
}

EthashAux* EthashAux::get()
{
    static std::once_flag flag;
    std::call_once(flag, []{s_this = new EthashAux();});
    return s_this;
}

uint64_t EthashAux::cacheSize(BlockInfo const& _header)
{
	return ethash_get_cachesize((uint64_t)_header.number());
}

uint64_t EthashAux::dataSize(uint64_t _blockNumber)
{
	return ethash_get_datasize(_blockNumber);
}


void EthashAux::setDAGDirName(const char * custom_dir_name)
{
	char strbuf[256];
	if (strcmp(custom_dir_name, "") != 0)
		strcpy(strbuf, custom_dir_name);
	else
		ethash_get_default_dirname(strbuf, 256);

	strcpy(s_dagDirName, strbuf);
}

char * EthashAux::dagDirName()
{
	return s_dagDirName;
}

void EthashAux::setDAGEraseMode(DAGEraseMode mode)
{
	s_dagEraseMode = mode;
}

void EthashAux::eraseDAGs()
{
	if (s_dagEraseMode == DAGEraseMode::None) return;

	path p(s_dagDirName);
	
	if (!is_directory(p))
	{
		cnote << "Can't clean up DAG directory:" << s_dagDirName << "does not exist (yet).";
		return;
	}

	vector<path> files;

	directory_iterator end_itr;

	for (directory_iterator itr(p); itr != end_itr; ++itr)
	{
		if (is_regular_file(itr->path())) {
			files.push_back(itr->path());
		}
	}

	std::sort(files.begin(), files.end(),
		[](const path& p1, const path& p2)
	{
		return file_size(p1) < file_size(p2);
	});

	size_t dagcount = files.size();
	size_t dagcounter = 0;
	for (std::vector<path>::const_iterator path = files.begin(); path != files.end(); ++path)
	{
		if (s_dagEraseMode == DAGEraseMode::Bench && path->filename() == "full-R23-0000000000000000") {}
		else if ((s_dagEraseMode == DAGEraseMode::Old || s_dagEraseMode == DAGEraseMode::Bench) && dagcount - dagcounter <= 2) {}
		else
		{
			cnote << "Deleting DAG file " << path->string();
			remove(*path);
		}
		dagcounter++;
	}
}

h256 EthashAux::seedHash(unsigned _number)
{
	unsigned epoch = _number / ETHASH_EPOCH_LENGTH;
	Guard l(get()->x_epochs);
	if (epoch >= get()->m_seedHashes.size())
	{
		h256 ret;
		unsigned n = 0;
		if (!get()->m_seedHashes.empty())
		{
			ret = get()->m_seedHashes.back();
			n = get()->m_seedHashes.size() - 1;
		}
		get()->m_seedHashes.resize(epoch + 1);
//		cdebug << "Searching for seedHash of epoch " << epoch;
		for (; n <= epoch; ++n, ret = sha3(ret))
		{
			get()->m_seedHashes[n] = ret;
//			cdebug << "Epoch" << n << "is" << ret;
		}
	}
	return get()->m_seedHashes[epoch];
}

uint64_t EthashAux::number(h256 const& _seedHash)
{
	Guard l(get()->x_epochs);
	unsigned epoch = 0;
	auto epochIter = get()->m_epochs.find(_seedHash);
	if (epochIter == get()->m_epochs.end())
	{
		//		cdebug << "Searching for seedHash " << _seedHash;
		for (h256 h; h != _seedHash && epoch < 2048; ++epoch, h = sha3(h), get()->m_epochs[h] = epoch) {}
		if (epoch == 2048)
		{
			std::ostringstream error;
			error << "apparent block number for " << _seedHash << " is too high; max is " << (ETHASH_EPOCH_LENGTH * 2048);
			throw std::invalid_argument(error.str());
		}
	}
	else
		epoch = epochIter->second;
	return epoch * ETHASH_EPOCH_LENGTH;
}

void EthashAux::killCache(h256 const& _s)
{
	WriteGuard l(x_lights);
	m_lights.erase(_s);
}

EthashAux::LightType EthashAux::light(h256 const& _seedHash)
{
	UpgradableGuard l(get()->x_lights);
	if (get()->m_lights.count(_seedHash))
		return get()->m_lights.at(_seedHash);
	UpgradeGuard l2(l);
	return (get()->m_lights[_seedHash] = make_shared<LightAllocation>(_seedHash));
}

EthashAux::LightAllocation::LightAllocation(h256 const& _seedHash)
{
	uint64_t blockNumber = EthashAux::number(_seedHash);
	light = ethash_light_new(blockNumber);
	if (!light)
		BOOST_THROW_EXCEPTION(ExternalFunctionFailure("ethash_light_new()"));
	size = ethash_get_cachesize(blockNumber);
}

EthashAux::LightAllocation::~LightAllocation()
{
	ethash_light_delete(light);
}

bytesConstRef EthashAux::LightAllocation::data() const
{
	return bytesConstRef((byte const*)light->cache, size);
}

EthashAux::FullAllocation::FullAllocation(ethash_light_t _light, ethash_callback_t _cb)
{
//	cdebug << "About to call ethash_full_new...";
	full = ethash_full_new(_light, EthashAux::dagDirName(), _cb);
//	cdebug << "Called OK.";
	if (!full)
	{
		clog(DAGChannel) << "DAG Generation Failure. Reason: "  << strerror(errno);
		BOOST_THROW_EXCEPTION(ExternalFunctionFailure("ethash_full_new"));
	}
}

EthashAux::FullAllocation::~FullAllocation()
{
	ethash_full_delete(full);
}

bytesConstRef EthashAux::FullAllocation::data() const
{
	return bytesConstRef((byte const*)ethash_full_dag(full), size());
}

static std::function<int(unsigned)> s_dagCallback;
static int dagCallbackShim(unsigned _p)
{
	clog(DAGChannel) << "Generating DAG file. Progress: " << toString(_p) << "%";
	return s_dagCallback ? s_dagCallback(_p) : 0;
}

EthashAux::FullType EthashAux::full(h256 const& _seedHash, bool _createIfMissing, function<int(unsigned)> const& _f)
{
	FullType ret;
	auto l = light(_seedHash);

	DEV_GUARDED(get()->x_fulls)
		if ((ret = get()->m_fulls[_seedHash].lock()))
		{
			get()->m_lastUsedFull = ret;
			return ret;
		}

	if (_createIfMissing || computeFull(_seedHash, false) == 100)
	{
		s_dagCallback = _f;
//		cnote << "Loading from libethash...";
		ret = make_shared<FullAllocation>(l->light, dagCallbackShim);
//		cnote << "Done loading.";

		DEV_GUARDED(get()->x_fulls)
			get()->m_fulls[_seedHash] = get()->m_lastUsedFull = ret;
	}

	return ret;
}

unsigned EthashAux::computeFull(h256 const& _seedHash, bool _createIfMissing)
{
	Guard l(get()->x_fulls);
	uint64_t blockNumber;

	DEV_IF_THROWS(blockNumber = EthashAux::number(_seedHash))
	{
		return 0;
	}

	if (FullType ret = get()->m_fulls[_seedHash].lock())
	{
		get()->m_lastUsedFull = ret;
		return 100;
	}

	if (_createIfMissing && (!get()->m_fullGenerator || !get()->m_fullGenerator->joinable()))
	{
		get()->m_fullProgress = 0;
		get()->m_generatingFullNumber = blockNumber / ETHASH_EPOCH_LENGTH * ETHASH_EPOCH_LENGTH;
		get()->m_fullGenerator = unique_ptr<thread>(new thread([=](){
			cnote << "Loading full DAG of seedhash: " << _seedHash;
			get()->full(_seedHash, true, [](unsigned p){ get()->m_fullProgress = p; return 0; });
			eraseDAGs();
			cnote << "Full DAG loaded";
			get()->m_fullProgress = 0;
			get()->m_generatingFullNumber = NotGenerating;
		}));
	}

	return (get()->m_generatingFullNumber == blockNumber) ? get()->m_fullProgress : 0;
}



EthashProofOfWork::Result EthashAux::FullAllocation::compute(h256 const& _headerHash, Nonce const& _nonce) const
{
	ethash_return_value_t r = ethash_full_compute(full, *(ethash_h256_t*)_headerHash.data(), (uint64_t)(u64)_nonce);
	if (!r.success)
		BOOST_THROW_EXCEPTION(DAGCreationFailure());
	return EthashProofOfWork::Result{h256((uint8_t*)&r.result, h256::ConstructFromPointer), h256((uint8_t*)&r.mix_hash, h256::ConstructFromPointer)};
}

EthashProofOfWork::Result EthashAux::LightAllocation::compute(h256 const& _headerHash, Nonce const& _nonce) const
{
	ethash_return_value r = ethash_light_compute(light, *(ethash_h256_t*)_headerHash.data(), (uint64_t)(u64)_nonce);
	if (!r.success)
		BOOST_THROW_EXCEPTION(DAGCreationFailure());
	return EthashProofOfWork::Result{h256((uint8_t*)&r.result, h256::ConstructFromPointer), h256((uint8_t*)&r.mix_hash, h256::ConstructFromPointer)};
}

EthashProofOfWork::Result EthashAux::eval(h256 const& _seedHash, h256 const& _headerHash, Nonce const& _nonce)
{
	DEV_GUARDED(get()->x_fulls)
		if (FullType dag = get()->m_fulls[_seedHash].lock())
			return dag->compute(_headerHash, _nonce);
	DEV_IF_THROWS(return EthashAux::get()->light(_seedHash)->compute(_headerHash, _nonce))
	{
		return EthashProofOfWork::Result{ ~h256(), h256() };
	}
}
