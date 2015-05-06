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
/** @file main.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 * Ethereum client.
 */
#if ETH_ETHASHCL
#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#include <libethash-cl/cl.hpp>
#pragma clang diagnostic pop
#else
#include <libethash-cl/cl.hpp>
#endif
#endif
#include <functional>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <libdevcore/RangeMask.h>
#include <libdevcore/Log.h>
#include <libdevcore/Common.h>
#include <libdevcore/CommonData.h>
#include <libdevcore/RLP.h>
#include <libdevcore/TransientDirectory.h>
#include <libdevcore/CommonIO.h>
#include <libdevcrypto/TrieDB.h>
#include <libp2p/All.h>
#include <libethcore/ProofOfWork.h>
#include <libdevcrypto/FileSystem.h>
#include <libethereum/All.h>
#include <libethereum/Farm.h>
#include <libethereum/AccountDiff.h>
#include <libethereum/DownloadMan.h>
#include <libethereum/Client.h>
#include <liblll/All.h>
#include <libwhisper/WhisperPeer.h>
#include <libwhisper/WhisperHost.h>
#include <test/JsonSpiritHeaders.h>
using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace dev::p2p;
using namespace dev::shh;
namespace js = json_spirit;
namespace fs = boost::filesystem;

#if 1

inline h128 fromUUID(std::string const& _uuid) { return h128(boost::replace_all_copy(_uuid, "-", "")); }

class KeyManager: public Worker
{
public:
	KeyManager() { readKeys(); }
	~KeyManager() {}

	Secret secret(h128 const& _uuid, std::string const& _pass)
	{
		auto it = m_keys.find(_uuid);
		if (it == m_keys.end())
			return Secret();
		return Secret(decrypt(it->second, _pass));
	}

private:
	void readKeys(std::string const& _keysPath = getDataDir("web3") + "/keys")
	{
		fs::path p(_keysPath);
		js::mValue v;
		for (fs::directory_iterator it(p); it != fs::directory_iterator(); ++it)
			if (is_regular_file(it->path()))
			{
				cdebug << "Reading" << it->path();
				js::read_string(contentsString(it->path().string()), v);
				js::mObject o = v.get_obj();
				int version = o.count("Version") ? stoi(o["Version"].get_str()) : o.count("version") ? o["version"].get_int() : 0;
				if (version == 2)
					m_keys[fromUUID(o["id"].get_str())] = o["crypto"];
				else
					cwarn << "Cannot read key version" << version;
			}
	}

	static bytes decrypt(js::mValue const& _v, std::string const& _pass)
	{
		js::mObject o = _v.get_obj();
		bytes pKey;
		if (o["kdf"].get_str() == "pbkdf2")
		{
			auto params = o["kdfparams"].get_obj();
			unsigned iterations = params["c"].get_int();
			bytes salt = fromHex(params["salt"].get_str());
			pKey = pbkdf2(_pass, salt, iterations).asBytes();
		}
		else
		{
			cwarn << "Unknown KDF" << o["kdf"].get_str() << "not supported.";
			return bytes();
		}

		// TODO check MAC
		h256 mac(o["mac"].get_str());
		(void)mac;

		bytes cipherText = fromHex(o["ciphertext"].get_str());
		bytes ret;
		if (o["cipher"].get_str() == "aes-128-cbc")
		{
			auto params = o["cipherparams"].get_obj();
			h128 key(sha3(h128(pKey, h128::AlignRight)), h128::AlignRight);
			h128 iv(params["iv"].get_str());
			decryptSymNoAuth(key, iv, &cipherText, ret);
		}
		else
		{
			cwarn << "Unknown cipher" << o["cipher"].get_str() << "not supported.";
			return bytes();
		}

		return ret;
	}

	std::map<h128, js::mValue> m_keys;
};

int main()
{
	KeyManager keyman;
	cdebug << "Secret key for 0498f19a-59db-4d54-ac95-33901b4f1870 is " << keyman.secret(fromUUID("0498f19a-59db-4d54-ac95-33901b4f1870"), "foo");
}

#elif 0
int main()
{
	DownloadMan man;
	DownloadSub s0(man);
	DownloadSub s1(man);
	DownloadSub s2(man);
	man.resetToChain(h256s({u256(0), u256(1), u256(2), u256(3), u256(4), u256(5), u256(6), u256(7), u256(8)}));
	assert((s0.nextFetch(2) == h256Set{(u256)7, (u256)8}));
	assert((s1.nextFetch(2) == h256Set{(u256)5, (u256)6}));
	assert((s2.nextFetch(2) == h256Set{(u256)3, (u256)4}));
	s0.noteBlock(u256(8));
	s0.doneFetch();
	assert((s0.nextFetch(2) == h256Set{(u256)2, (u256)7}));
	s1.noteBlock(u256(6));
	s1.noteBlock(u256(5));
	s1.doneFetch();
	assert((s1.nextFetch(2) == h256Set{(u256)0, (u256)1}));
	s0.doneFetch();				// TODO: check exact semantics of doneFetch & nextFetch. Not sure if they're right -> doneFetch calls resetFetch which kills all the info of past fetches.
	cdebug << s0.nextFetch(2);
	assert((s0.nextFetch(2) == h256Set{(u256)3, (u256)4}));

/*	RangeMask<unsigned> m(0, 100);
	cnote << m;
	m += UnsignedRange(3, 10);
	cnote << m;
	m += UnsignedRange(11, 16);
	cnote << m;
	m += UnsignedRange(10, 11);
	cnote << m;
	cnote << ~m;
	cnote << (~m).lowest(10);
	for (auto i: (~m).lowest(10))
		cnote << i;*/
	return 0;
}
#elif 0
int main()
{
	KeyPair u = KeyPair::create();
	KeyPair cb = KeyPair::create();
	OverlayDB db;
	State s(cb.address(), db, BaseState::Empty);
	cnote << s.rootHash();
	s.addBalance(u.address(), 1 * ether);
	Address c = s.newContract(1000 * ether, compileLLL("(suicide (caller))"));
	s.commit();
	State before = s;
	cnote << "State before transaction: " << before;
	Transaction t(0, 10000, 10000, c, bytes(), 0, u.secret());
	cnote << "Transaction: " << t;
	cnote << s.balance(c);
	s.execute(LastHashes(), t.rlp());
	cnote << "State after transaction: " << s;
	cnote << before.diff(s);
}
#elif 0
int main()
{
	GenericFarm<Ethash> f;
	BlockInfo genesis = CanonBlockChain::genesis();
	genesis.difficulty = 1 << 18;
	cdebug << genesis.boundary();

	auto mine = [](GenericFarm<Ethash>& f, BlockInfo const& g, unsigned timeout) {
		BlockInfo bi = g;
		bool completed = false;
		f.onSolutionFound([&](ProofOfWork::Solution sol)
		{
			ProofOfWork::assignResult(sol, bi);
			return completed = true;
		});
		f.setWork(bi);
		for (unsigned i = 0; !completed && i < timeout * 10; ++i, cout << f.miningProgress() << "\r" << flush)
			this_thread::sleep_for(chrono::milliseconds(100));
		cout << endl << flush;
		cdebug << bi.mixHash << bi.nonce << (Ethash::verify(bi) ? "GOOD" : "bad");
	};

	Ethash::prep(genesis);

	genesis.difficulty = u256(1) << 40;
	genesis.noteDirty();
	f.startCPU();
	mine(f, genesis, 10);

	f.startGPU();

	cdebug << "Good:";
	genesis.difficulty = 1 << 18;
	genesis.noteDirty();
	mine(f, genesis, 30);

	cdebug << "Bad:";
	genesis.difficulty = (u256(1) << 40);
	genesis.noteDirty();
	mine(f, genesis, 30);

	f.stop();

	return 0;
}
#elif 0

void mine(State& s, BlockChain const& _bc)
{
	s.commitToMine(_bc);
	GenericFarm<ProofOfWork> f;
	bool completed = false;
	f.onSolutionFound([&](ProofOfWork::Solution sol)
	{
		return completed = s.completeMine<ProofOfWork>(sol);
	});
	f.setWork(s.info());
	f.startCPU();
	while (!completed)
		this_thread::sleep_for(chrono::milliseconds(20));
}
#elif 0
int main()
{
	cnote << "Testing State...";

	KeyPair me = sha3("Gav Wood");
	KeyPair myMiner = sha3("Gav's Miner");
//	KeyPair you = sha3("123");

	Defaults::setDBPath(boost::filesystem::temp_directory_path().string() + "/" + toString(chrono::system_clock::now().time_since_epoch().count()));

	OverlayDB stateDB = State::openDB();
	CanonBlockChain bc;
	cout << bc;

	State s(stateDB, BaseState::CanonGenesis, myMiner.address());
	cout << s;

	// Sync up - this won't do much until we use the last state.
	s.sync(bc);

	cout << s;

	// Mine to get some ether!
	mine(s, bc);

	bc.attemptImport(s.blockData(), stateDB);

	cout << bc;

	s.sync(bc);

	cout << s;

	// Inject a transaction to transfer funds from miner to me.
	Transaction t(1000, 10000, 30000, me.address(), bytes(), s.transactionsFrom(myMiner.address()), myMiner.secret());
	assert(t.sender() == myMiner.address());
	s.execute(bc.lastHashes(), t);

	cout << s;

	// Mine to get some ether and set in stone.
	s.commitToMine(bc);
	s.commitToMine(bc);
	mine(s, bc);
	bc.attemptImport(s.blockData(), stateDB);

	cout << bc;

	s.sync(bc);

	cout << s;

	return 0;
}
#else
int main()
{
	string tempDir = boost::filesystem::temp_directory_path().string() + "/" + toString(chrono::system_clock::now().time_since_epoch().count());

	KeyPair myMiner = sha3("Gav's Miner");

	p2p::Host net("Test");
	cdebug << "Path:" << tempDir;
	Client c(&net, tempDir);

	c.setAddress(myMiner.address());

	this_thread::sleep_for(chrono::milliseconds(1000));

	c.startMining();

	this_thread::sleep_for(chrono::milliseconds(6000));

	c.stopMining();

	return 0;
}
#endif

