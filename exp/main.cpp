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
inline std::string toUUID(h128 const& _uuid) { std::string ret = toHex(_uuid.ref()); for (unsigned i: {20, 16, 12, 8}) ret.insert(ret.begin() + i, '-'); return ret; }

class KeyStore
{
public:
	KeyStore() { readKeys(); }
	~KeyStore() {}

	bytes key(h128 const& _uuid, function<std::string()> const& _pass)
	{
		auto rit = m_cached.find(_uuid);
		if (rit != m_cached.end())
			return rit->second;
		auto it = m_keys.find(_uuid);
		if (it == m_keys.end())
			return bytes();
		bytes key = decrypt(it->second, _pass());
		if (!key.empty())
			m_cached[_uuid] = key;
		return key;
	}

	h128 import(bytes const& _s, std::string const& _pass)
	{
		h128 r = h128::random();
		m_cached[r] = _s;
		m_keys[r] = encrypt(_s, _pass);
		writeKeys();
		return r;
	}

	// Clear any cached keys.
	void clearCache() const { m_cached.clear(); }

private:
	void writeKeys(std::string const& _keysPath = getDataDir("web3") + "/keys")
	{
		fs::path p(_keysPath);
		boost::filesystem::create_directories(p);
		for (auto const& k: m_keys)
		{
			std::string uuid = toUUID(k.first);
			js::mObject v;
			v["crypto"] = k.second;
			v["id"] = uuid;
			v["version"] = 2;
			writeFile((p / uuid).string() + ".json", js::write_string(js::mValue(v), true));
		}
	}

	void readKeys(std::string const& _keysPath = getDataDir("web3") + "/keys")
	{
		fs::path p(_keysPath);
		js::mValue v;
		for (fs::directory_iterator it(p); it != fs::directory_iterator(); ++it)
			if (is_regular_file(it->path()))
			{
				cdebug << "Reading" << it->path();
				js::read_string(contentsString(it->path().string()), v);
				if (v.type() == js::obj_type)
				{
					js::mObject o = v.get_obj();
					int version = o.count("Version") ? stoi(o["Version"].get_str()) : o.count("version") ? o["version"].get_int() : 0;
					if (version == 2)
						m_keys[fromUUID(o["id"].get_str())] = o["crypto"];
					else
						cwarn << "Cannot read key version" << version;
				}
//				else
//					cwarn << "Invalid JSON in key file" << it->path().string();
			}
	}

	static js::mValue encrypt(bytes const& _v, std::string const& _pass)
	{
		js::mObject ret;

		// KDF info
		unsigned dklen = 16;
		unsigned iterations = 262144;
		bytes salt = h256::random().asBytes();
		ret["kdf"] = "pbkdf2";
		{
			js::mObject params;
			params["prf"] = "hmac-sha256";
			params["c"] = (int)iterations;
			params["salt"] = toHex(salt);
			params["dklen"] = (int)dklen;
			ret["kdfparams"] = params;
		}
		bytes derivedKey = pbkdf2(_pass, salt, iterations, dklen);

		// cipher info
		ret["cipher"] = "aes-128-cbc";
		h128 key(sha3(h128(derivedKey, h128::AlignRight)), h128::AlignRight);
		h128 iv = h128::random();
		{
			js::mObject params;
			params["iv"] = toHex(iv.ref());
			ret["cipherparams"] = params;
		}

		// cipher text
		bytes cipherText = encryptSymNoAuth(key, iv, &_v);
		ret["ciphertext"] = toHex(cipherText);

		// and mac.
		h256 mac = sha3(bytesConstRef(&derivedKey).cropped(derivedKey.size() - 16).toBytes() + cipherText);
		ret["mac"] = toHex(mac.ref());

		return ret;
	}

	static bytes decrypt(js::mValue const& _v, std::string const& _pass)
	{
		js::mObject o = _v.get_obj();

		// derive key
		bytes derivedKey;
		if (o["kdf"].get_str() == "pbkdf2")
		{
			auto params = o["kdfparams"].get_obj();
			if (params["prf"].get_str() != "hmac-sha256")
			{
				cwarn << "Unknown PRF for PBKDF2" << params["prf"].get_str() << "not supported.";
				return bytes();
			}
			unsigned iterations = params["c"].get_int();
			bytes salt = fromHex(params["salt"].get_str());
			derivedKey = pbkdf2(_pass, salt, iterations, params["dklen"].get_int());
		}
		else
		{
			cwarn << "Unknown KDF" << o["kdf"].get_str() << "not supported.";
			return bytes();
		}

		bytes cipherText = fromHex(o["ciphertext"].get_str());

		// check MAC
		h256 mac(o["mac"].get_str());
		h256 macExp = sha3(bytesConstRef(&derivedKey).cropped(derivedKey.size() - 16).toBytes() + cipherText);
		if (mac != macExp)
		{
			cwarn << "Invalid key - MAC mismatch; expected" << toString(macExp) << ", got" << toString(mac);
			return bytes();
		}

		// decrypt
		if (o["cipher"].get_str() == "aes-128-cbc")
		{
			auto params = o["cipherparams"].get_obj();
			h128 key(sha3(h128(derivedKey, h128::AlignRight)), h128::AlignRight);
			h128 iv(params["iv"].get_str());
			return decryptSymNoAuth(key, iv, &cipherText);
		}
		else
		{
			cwarn << "Unknown cipher" << o["cipher"].get_str() << "not supported.";
			return bytes();
		}
	}

	mutable std::map<h128, bytes> m_cached;
	std::map<h128, js::mValue> m_keys;
};

class UnknownPassword: public Exception {};

struct KeyInfo
{
	h256 passHash;
	std::string name;
};

static const auto DontKnowThrow = [](){ BOOST_THROW_EXCEPTION(UnknownPassword()); return std::string(); };

// TODO: This one is specifically for Ethereum, but we can make it generic in due course.
// TODO: hidden-partition style key-store.
/**
 * @brief High-level manager of keys for Ethereum.
 * Usage:
 *
 * Call exists() to check whether there is already a database. If so, get the master password from
 * the user and call load() with it. If not, get a new master password from the user (get them to type
 * it twice and keep some hint around!) and call create() with it.
 */
class KeyManager
{
public:
	KeyManager() {}
	~KeyManager() {}

	void setKeysFile(std::string const& _keysFile) { m_keysFile = _keysFile; }
	std::string const& keysFile() const { return m_keysFile; }

	bool exists()
	{
		return !contents(m_keysFile + ".salt").empty() && !contents(m_keysFile).empty();
	}

	void create(std::string const& _pass)
	{
		m_password = asString(h256::random().asBytes());
		save(_pass, m_keysFile);
	}

	bool load(std::string const& _pass)
	{
		try {
			bytes salt = contents(m_keysFile + ".salt");
			bytes encKeys = contents(m_keysFile);
			m_key = h128(pbkdf2(_pass, salt, 262144, 16));
			bytes bs = decryptSymNoAuth(m_key, h128(), &encKeys);
			RLP s(bs);
			unsigned version = (unsigned)s[0];
			if (version == 1)
			{
				for (auto const& i: s[1])
					m_keyInfo[m_addrLookup[(Address)i[0]] = (h128)i[1]] = KeyInfo{(h256)i[2], (std::string)i[3]};
				for (auto const& i: s[2])
					m_passwordInfo[(h256)i[0]] = (std::string)i[1];
				m_password = (string)s[3];
			}
			m_cachedPasswords[sha3(m_password)] = m_password;
			return true;
		}
		catch (...) {
			return false;
		}
	}

	void resave(std::string const& _pass)
	{
		save(_pass, m_keysFile);
	}

	Secret secret(Address const& _address, function<std::string()> const& _pass = DontKnowThrow)
	{
		auto it = m_addrLookup.find(_address);
		if (it == m_addrLookup.end())
			return Secret();
		return secret(it->second, _pass);
	}

	Secret secret(h128 const& _uuid, function<std::string()> const& _pass = DontKnowThrow)
	{
		return Secret(m_store.key(_uuid, [&](){
			auto it = m_cachedPasswords.find(m_keyInfo[_uuid].passHash);
			if (it == m_cachedPasswords.end())
			{
				std::string p = _pass();
				m_cachedPasswords[sha3(p)] = p;
				return p;
			}
			else
				return it->second;
		}));
	}

	h128 import(Secret const& _s, std::string const& _pass, string const& _info = std::string(), string const& _passInfo = std::string())
	{
		Address addr = KeyPair(_s).address();
		auto passHash = sha3(_pass);
		m_cachedPasswords[passHash] = _pass;
		m_passwordInfo[passHash] = _passInfo;
		auto uuid = m_store.import(_s.asBytes(), _pass);
		m_keyInfo[uuid] = KeyInfo{passHash, _info};
		m_addrLookup[addr] = uuid;
		save(m_keysFile);
		return uuid;
	}

	h128 import(Secret const& _s, std::string const& _info = std::string())
	{
		// cache password, remember the key, remember the address
		return import(_s, m_password, _info, std::string());
	}

	void importExisting(h128 const& _uuid, std::string const& _pass, std::string const& _info = std::string(), std::string const& _passInfo = std::string())
	{
		bytes key = m_store.key(_uuid, [&](){ return _pass; });
		if (key.empty())
			return;
		Address a = KeyPair(Secret(key)).address();
		auto passHash = sha3(_pass);
		if (!m_passwordInfo.count(passHash))
			m_passwordInfo[passHash] = _passInfo;
		if (!m_cachedPasswords.count(passHash))
			m_cachedPasswords[passHash] = _pass;
		m_addrLookup[a] = _uuid;
		m_keyInfo[_uuid].passHash = passHash;
		m_keyInfo[_uuid].name = _info;
		save(m_keysFile);
	}

	KeyStore& store() { return m_store; }

private:
	// Only use if previously loaded ok.
	// @returns false if wasn't previously loaded ok.
	bool save(std::string const& _keysFile)
	{
		if (!m_key)
			return false;
		save(m_key, _keysFile);
		return true;
	}

	void save(std::string const& _pass, std::string const& _keysFile)
	{
		bytes salt = h256::random().asBytes();
		writeFile(_keysFile + ".salt", salt);
		auto key = h128(pbkdf2(_pass, salt, 262144, 16));
		save(key, _keysFile);
	}

	void save(h128 const& _key, std::string const& _keysFile)
	{
		RLPStream s(4);
		s << 1;
		s.appendList(m_addrLookup.size());
		for (auto const& i: m_addrLookup)
			s.appendList(4) << i.first << i.second << m_keyInfo[i.second].passHash << m_keyInfo[i.second].name;
		s.appendList(m_passwordInfo.size());
		for (auto const& i: m_passwordInfo)
			s.appendList(2) << i.first << i.second;
		s.append(m_password);

		writeFile(_keysFile, encryptSymNoAuth(_key, h128(), &s.out()));
		m_key = _key;
	}

	// Ethereum keys.
	std::map<Address, h128> m_addrLookup;
	std::map<h128, KeyInfo> m_keyInfo;
	std::map<h256, std::string> m_passwordInfo;

	// Passwords that we're storing.
	std::map<h256, std::string> m_cachedPasswords;

	// The default password for keys in the keystore - protected by the master password.
	std::string m_password;

	KeyStore m_store;
	h128 m_key;
	std::string m_keysFile = getDataDir("ethereum") + "/keys.info";
};

int main()
{
	KeyManager keyman;
	if (keyman.exists())
		keyman.load("foo");
	else
		keyman.create("foo");

	auto id = fromUUID("441193ae-a767-f1c3-48ba-dd6610db5ed0");
	keyman.importExisting(id, "bar");

	cdebug << "Secret key for " << toUUID(id) << "is" << keyman.store().key(id, [](){ return "bar"; });
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

