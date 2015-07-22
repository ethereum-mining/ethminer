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
#if 0
#include <libdevcore/TrieDB.h>
#include <libdevcore/TrieHash.h>
#include <libdevcore/RangeMask.h>
#include <libdevcore/Log.h>
#include <libdevcore/Common.h>
#include <libdevcore/CommonData.h>
#include <libdevcore/RLP.h>
#include <libdevcore/TransientDirectory.h>
#include <libdevcore/CommonIO.h>
#include <libdevcrypto/SecretStore.h>
#include <libp2p/All.h>
#include <libethcore/Farm.h>
#include <libdevcore/FileSystem.h>
#include <libethereum/All.h>
#include <libethcore/KeyManager.h>
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
#else
#include <libethcore/Sealer.h>
#include <libethcore/BasicAuthority.h>
#include <libethcore/BlockInfo.h>
#include <libethcore/Ethash.h>
#include <libethcore/Params.h>
#include <libethereum/All.h>
#include <libethereum/AccountDiff.h>
#include <libethereum/DownloadMan.h>
#include <libethereum/Client.h>
using namespace std;
using namespace dev;
using namespace eth;
#endif

#if 0
int main()
{
	BlockInfo bi;
	bi.difficulty = c_genesisDifficulty;
	bi.gasLimit = c_genesisGasLimit;
	bi.number() = 1;
	bi.parentHash() = sha3("parentHash");

	bytes sealedData;

	{
		KeyPair kp(sha3("test"));
		SealEngineFace* se = BasicAuthority::createSealEngine();
		se->setOption("authority", rlp(kp.secret()));
		se->setOption("authorities", rlpList(kp.address()));
		cdebug << se->sealers();
		bool done = false;
		se->onSealGenerated([&](SealFace const* seal){
			sealedData = seal->sealedHeader(bi);
			done = true;
		});
		se->generateSeal(bi);
		while (!done)
			this_thread::sleep_for(chrono::milliseconds(50));
		BasicAuthority::BlockHeader sealed = BasicAuthority::BlockHeader::fromHeader(sealedData, CheckEverything);
		cdebug << sealed.sig();
	}

	{
		SealEngineFace* se = Ethash::createSealEngine();
		cdebug << se->sealers();
		bool done = false;
		se->setSealer("cpu");
		se->onSealGenerated([&](SealFace const* seal){
			sealedData = seal->sealedHeader(bi);
			done = true;
		});
		se->generateSeal(bi);
		while (!done)
			this_thread::sleep_for(chrono::milliseconds(50));
		Ethash::BlockHeader sealed = Ethash::BlockHeader::fromHeader(sealedData, CheckEverything);
		cdebug << sealed.nonce();
	}

	return 0;
}
#elif 0
int main()
{
	cdebug << pbkdf2("password", asBytes("salt"), 1, 32);
	cdebug << pbkdf2("password", asBytes("salt"), 1, 16);
	cdebug << pbkdf2("password", asBytes("salt"), 2, 16);
	cdebug << pbkdf2("testpassword", fromHex("de5742f1f1045c402296422cee5a8a9ecf0ac5bf594deca1170d22aef33a79cf"), 262144, 16);
	return 0;
}
#elif 0
int main()
{
	cdebug << "EXP";
	vector<bytes> data;
	for (unsigned i = 0; i < 10000; ++i)
		data.push_back(rlp(i));

	h256 ret;
	DEV_TIMED("triedb")
	{
		MemoryDB mdb;
		GenericTrieDB<MemoryDB> t(&mdb);
		t.init();
		unsigned i = 0;
		for (auto const& d: data)
			t.insert(rlp(i++), d);
		ret = t.root();
	}
	cdebug << ret;
	DEV_TIMED("hash256")
		ret = orderedTrieRoot(data);
	cdebug << ret;
}
#elif 0
int main()
{
	KeyManager keyman;
	if (keyman.exists())
		keyman.load("foo");
	else
		keyman.create("foo");

	Address a("9cab1cc4e8fe528267c6c3af664a1adbce810b5f");

//	keyman.importExisting(fromUUID("441193ae-a767-f1c3-48ba-dd6610db5ed0"), "{\"name\":\"Gavin Wood - Main identity\"}", "bar", "{\"hint\":\"Not foo.\"}");
//	Address a2 = keyman.address(keyman.import(Secret::random(), "Key with no additional security."));
//	cdebug << toString(a2);
	Address a2("19c486071651b2650449ba3c6a807f316a73e8fe");

	cdebug << keyman.accountDetails();

	cdebug << "Secret key for " << a << "is" << keyman.secret(a, [](){ return "bar"; });
	cdebug << "Secret key for " << a2 << "is" << keyman.secret(a2);

}
#elif 0
int main()
{
	DownloadMan man;
	DownloadSub s0(man);
	DownloadSub s1(man);
	DownloadSub s2(man);
	man.resetToChain(h256s({u256(0), u256(1), u256(2), u256(3), u256(4), u256(5), u256(6), u256(7), u256(8)}), 0);
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
	GenericFarm<EthashProofOfWork> f;
	BlockInfo genesis = CanonBlockChain::genesis();
	genesis.difficulty = 1 << 18;
	cdebug << genesis.boundary();

	auto mine = [](GenericFarm<EthashProofOfWork>& f, BlockInfo const& g, unsigned timeout) {
		BlockInfo bi = g;
		bool completed = false;
		f.onSolutionFound([&](EthashProofOfWork::Solution sol)
		{
			bi.proof = sol;
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
#elif 1
int main()
{
	bytes tx = fromHex("f84c01028332dcd58004801ba024843272ee176277535489859cbd275686023fe64aabd158b6fcdf2ae6a1ab6ba02f252a5016a48e5ec8d17aefaf4324d29b9e123fa623dc5a60539b3ad3610c95");
	Transaction t(tx, CheckTransaction::None);
	Public p = recover(t.signature(), t.sha3(WithoutSignature));
	cnote << t.signature().r;
	cnote << t.signature().s;
	cnote << t.signature().v;
	cnote << p;
	cnote << toAddress(p);
	cnote << t.sender();
}
#elif 0
void mine(State& s, BlockChain const& _bc, SealEngineFace* _se)
{
	s.commitToSeal(_bc);
	Notified<bytes> sealed;
	_se->onSealGenerated([&](bytes const& sealedHeader){ sealed = sealedHeader; });
	_se->generateSeal(s.info());
	sealed.waitNot({});
	s.sealBlock(sealed);
}
int main()
{
	cnote << "Testing State...";

	KeyPair me = sha3("Gav Wood");
	KeyPair myMiner = sha3("Gav's Miner");
//	KeyPair you = sha3("123");

	Defaults::setDBPath(boost::filesystem::temp_directory_path().string() + "/" + toString(chrono::system_clock::now().time_since_epoch().count()));

	using Sealer = Ethash;
	CanonBlockChain<Sealer> bc;
	auto gbb = bc.headerData(bc.genesisHash());
	assert(Sealer::BlockHeader(bc.headerData(bc.genesisHash()), IgnoreSeal, bc.genesisHash(), HeaderData));

	SealEngineFace* se = Sealer::createSealEngine();
	KeyPair kp(sha3("test"));
	se->setOption("authority", rlp(kp.secret()));
	se->setOption("authorities", rlpList(kp.address()));

	OverlayDB stateDB = State::openDB(bc.genesisHash());
	cnote << bc;

	Block s = bc.genesisBlock(stateDB);
	s.setBeneficiary(myMiner.address());
	cnote << s;

	// Sync up - this won't do much until we use the last state.
	s.sync(bc);

	cnote << s;

	// Mine to get some ether!
	mine(s, bc, se);

	bytes minedBlock = s.blockData();
	cnote << "Mined block is" << BlockInfo(minedBlock).stateRoot();
	bc.import(minedBlock, stateDB);

	cnote << bc;

	s.sync(bc);

	cnote << s;
	cnote << "Miner now has" << s.balance(myMiner.address());
	s.resetCurrent();
	cnote << "Miner now has" << s.balance(myMiner.address());

	// Inject a transaction to transfer funds from miner to me.
	Transaction t(1000, 10000, 30000, me.address(), bytes(), s.transactionsFrom(myMiner.address()), myMiner.secret());
	assert(t.sender() == myMiner.address());
	s.execute(bc.lastHashes(), t);

	cnote << s;

	// Mine to get some ether and set in stone.
	s.commitToSeal(bc);
	s.commitToSeal(bc);
	mine(s, bc, se);
	bc.attemptImport(s.blockData(), stateDB);

	cnote << bc;

	s.sync(bc);

	cnote << s;

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

	c.setBeneficiary(myMiner.address());

	this_thread::sleep_for(chrono::milliseconds(1000));

	c.startMining();

	this_thread::sleep_for(chrono::milliseconds(6000));

	c.stopMining();

	return 0;
}
#endif

