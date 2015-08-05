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
/** @file KeyManager.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "KeyManager.h"
#include <thread>
#include <mutex>
#include <boost/filesystem.hpp>
#include <test/JsonSpiritHeaders.h>
#include <libdevcore/Log.h>
#include <libdevcore/Guards.h>
#include <libdevcore/RLP.h>
using namespace std;
using namespace dev;
using namespace eth;
namespace js = json_spirit;
namespace fs = boost::filesystem;

KeyManager::KeyManager(string const& _keysFile, string const& _secretsPath):
	m_keysFile(_keysFile), m_store(_secretsPath)
{}

KeyManager::~KeyManager()
{}

bool KeyManager::exists() const
{
	return !contents(m_keysFile + ".salt").empty() && !contents(m_keysFile).empty();
}

void KeyManager::create(string const& _pass)
{
	m_defaultPasswordDeprecated = asString(h256::random().asBytes());
	write(_pass, m_keysFile);
}

bool KeyManager::recode(Address const& _address, string const& _newPass, string const& _hint, function<string()> const& _pass, KDF _kdf)
{
	noteHint(_newPass, _hint);
	h128 u = uuid(_address);
	if (!store().recode(u, _newPass, [&](){ return getPassword(u, _pass); }, _kdf))
		return false;

	m_keyInfo[u].passHash = hashPassword(_newPass);
	write();
	return true;
}

bool KeyManager::recode(Address const& _address, SemanticPassword _newPass, function<string()> const& _pass, KDF _kdf)
{
	h128 u = uuid(_address);
	string p;
	if (_newPass == SemanticPassword::Existing)
		p = getPassword(u, _pass);
	else if (_newPass == SemanticPassword::Master)
		p = defaultPassword();
	else
		return false;

	return recode(_address, p, string(), _pass, _kdf);
}

bool KeyManager::load(string const& _pass)
{
	try
	{
		bytes salt = contents(m_keysFile + ".salt");
		bytes encKeys = contents(m_keysFile);
		if (encKeys.empty())
			return false;
		m_keysFileKey = SecureFixedHash<16>(pbkdf2(_pass, salt, 262144, 16));
		bytesSec bs = decryptSymNoAuth(m_keysFileKey, h128(), &encKeys);
		RLP s(bs.ref());
		unsigned version = unsigned(s[0]);
		if (version == 1)
		{
			for (auto const& i: s[1])
			{
				h128 uuid(i[1]);
				Address addr(i[0]);
				m_addrLookup[addr] = uuid;
				m_keyInfo[uuid] = KeyInfo(h256(i[2]), string(i[3]));
//				cdebug << toString(addr) << toString(uuid) << toString((h256)i[2]) << (string)i[3];
			}

			for (auto const& i: s[2])
				m_passwordHint[h256(i[0])] = string(i[1]);
			m_defaultPasswordDeprecated = string(s[3]);
		}
//		cdebug << hashPassword(m_password) << toHex(m_password);
		cachePassword(m_defaultPasswordDeprecated);
//		cdebug << hashPassword(asString(m_key.ref())) << m_key.hex();
		cachePassword(asString(m_keysFileKey.ref()));
//		cdebug << hashPassword(_pass) << _pass;
		m_master = hashPassword(_pass);
		cachePassword(_pass);
		return true;
	}
	catch (...)
	{
		return false;
	}
}

Secret KeyManager::secret(Address const& _address, function<string()> const& _pass) const
{
	auto it = m_addrLookup.find(_address);
	if (it == m_addrLookup.end())
		return Secret();
	return secret(it->second, _pass);
}

Secret KeyManager::secret(h128 const& _uuid, function<string()> const& _pass) const
{
	return Secret(m_store.secret(_uuid, [&](){ return getPassword(_uuid, _pass); }));
}

string KeyManager::getPassword(h128 const& _uuid, function<string()> const& _pass) const
{
	auto kit = m_keyInfo.find(_uuid);
	h256 ph;
	if (kit != m_keyInfo.end())
		ph = kit->second.passHash;
	return getPassword(ph, _pass);
}

string KeyManager::getPassword(h256 const& _passHash, function<string()> const& _pass) const
{
	auto it = m_cachedPasswords.find(_passHash);
	if (it != m_cachedPasswords.end())
		return it->second;
	for (unsigned i = 0; i < 10; ++i)
	{
		string p = _pass();
		if (p.empty())
			break;
		if (_passHash == UnknownPassword || hashPassword(p) == _passHash)
		{
			cachePassword(p);
			return p;
		}
	}
	return string();
}

h128 KeyManager::uuid(Address const& _a) const
{
	auto it = m_addrLookup.find(_a);
	if (it == m_addrLookup.end())
		return h128();
	return it->second;
}

Address KeyManager::address(h128 const& _uuid) const
{
	for (auto const& i: m_addrLookup)
		if (i.second == _uuid)
			return i.first;
	return Address();
}

h128 KeyManager::import(Secret const& _s, string const& _accountName, string const& _pass, string const& _passwordHint)
{
	Address addr = KeyPair(_s).address();
	auto passHash = hashPassword(_pass);
	cachePassword(_pass);
	m_passwordHint[passHash] = _passwordHint;
	auto uuid = m_store.importSecret(_s.asBytesSec(), _pass);
	m_keyInfo[uuid] = KeyInfo{passHash, _accountName};
	m_addrLookup[addr] = uuid;
	write(m_keysFile);
	return uuid;
}

void KeyManager::importExisting(h128 const& _uuid, string const& _info, string const& _pass, string const& _passwordHint)
{
	bytesSec key = m_store.secret(_uuid, [&](){ return _pass; });
	if (key.empty())
		return;
	Address a = KeyPair(Secret(key)).address();
	auto passHash = hashPassword(_pass);
	if (!m_cachedPasswords.count(passHash))
		cachePassword(_pass);
	importExisting(_uuid, _info, a, passHash, _passwordHint);
}

void KeyManager::importExisting(h128 const& _uuid, string const& _accountName, Address const& _address, h256 const& _passHash, string const& _passwordHint)
{
	if (!m_passwordHint.count(_passHash))
		m_passwordHint[_passHash] = _passwordHint;
	m_addrLookup[_address] = _uuid;
	m_keyInfo[_uuid].passHash = _passHash;
	m_keyInfo[_uuid].accountName = _accountName;
	write(m_keysFile);
}

void KeyManager::kill(Address const& _a)
{
	auto id = m_addrLookup[_a];
	m_addrLookup.erase(_a);
	m_keyInfo.erase(id);
	m_store.kill(id);
	write(m_keysFile);
}

KeyPair KeyManager::presaleSecret(std::string const& _json, function<string(bool)> const& _password)
{
	js::mValue val;
	json_spirit::read_string(_json, val);
	auto obj = val.get_obj();
	string p = _password(true);
	if (obj["encseed"].type() == js::str_type)
	{
		auto encseed = fromHex(obj["encseed"].get_str());
		KeyPair k;
		for (bool gotit = false; !gotit;)
		{
			gotit = true;
			k = KeyPair::fromEncryptedSeed(&encseed, p);
			if (obj["ethaddr"].type() == js::str_type)
			{
				Address a(obj["ethaddr"].get_str());
				Address b = k.address();
				if (a != b)
				{
					if ((p = _password(false)).empty())
						BOOST_THROW_EXCEPTION(PasswordUnknown());
					else
						gotit = false;
				}
			}
		}
		return k;
	}
	else
		BOOST_THROW_EXCEPTION(Exception() << errinfo_comment("encseed type is not js::str_type"));
}

Addresses KeyManager::accounts() const
{
	Addresses ret;
	ret.reserve(m_addrLookup.size());
	for (auto const& i: m_addrLookup)
		if (m_keyInfo.count(i.second) > 0)
			ret.push_back(i.first);
	return ret;
}

bool KeyManager::hasAccount(const Address& _address) const
{
	return m_addrLookup.count(_address) && m_keyInfo.count(m_addrLookup.at(_address));
}

string const& KeyManager::accountName(Address const& _address) const
{
	try
	{
		return m_keyInfo.at(m_addrLookup.at(_address)).accountName;
	}
	catch (...)
	{
		return EmptyString;
	}
}

string const& KeyManager::passwordHint(Address const& _address) const
{
	try
	{
		return m_passwordHint.at(m_keyInfo.at(m_addrLookup.at(_address)).passHash);
	}
	catch (...)
	{
		return EmptyString;
	}
}

h256 KeyManager::hashPassword(string const& _pass) const
{
	// TODO SECURITY: store this a bit more securely; Scrypt perhaps?
	return h256(pbkdf2(_pass, asBytes(m_defaultPasswordDeprecated), 262144, 32).makeInsecure());
}

void KeyManager::cachePassword(string const& _password) const
{
	m_cachedPasswords[hashPassword(_password)] = _password;
}

bool KeyManager::write(string const& _keysFile) const
{
	if (!m_keysFileKey)
		return false;
	write(m_keysFileKey, _keysFile);
	return true;
}

void KeyManager::write(string const& _pass, string const& _keysFile) const
{
	bytes salt = h256::random().asBytes();
	writeFile(_keysFile + ".salt", salt, true);
	auto key = SecureFixedHash<16>(pbkdf2(_pass, salt, 262144, 16));

	cachePassword(_pass);
	m_master = hashPassword(_pass);
	write(key, _keysFile);
}

void KeyManager::write(SecureFixedHash<16> const& _key, string const& _keysFile) const
{
	RLPStream s(4);
	s << 1; // version
	s.appendList(accounts().size());
	for (auto const& address: accounts())
	{
		h128 id = uuid(address);
		auto const& ki = m_keyInfo.at(id);
		s.appendList(4) << address << id << ki.passHash << ki.accountName;
	}
	s.appendList(m_passwordHint.size());
	for (auto const& i: m_passwordHint)
		s.appendList(2) << i.first << i.second;
	s.append(m_defaultPasswordDeprecated);

	writeFile(_keysFile, encryptSymNoAuth(_key, h128(), &s.out()), true);
	m_keysFileKey = _key;
	cachePassword(defaultPassword());
}
