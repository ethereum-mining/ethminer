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
#include <libdevcore/Log.h>
#include <libdevcore/Guards.h>
#include <libdevcore/RLP.h>
using namespace std;
using namespace dev;
using namespace eth;
namespace fs = boost::filesystem;

KeyManager::KeyManager(std::string const& _keysFile):
	m_keysFile(_keysFile)
{}

KeyManager::~KeyManager()
{}

bool KeyManager::exists() const
{
	return !contents(m_keysFile + ".salt").empty() && !contents(m_keysFile).empty();
}

void KeyManager::create(std::string const& _pass)
{
	m_password = asString(h256::random().asBytes());
	write(_pass, m_keysFile);
}

void KeyManager::reencode(Address const& _address, std::function<string()> const& _pass, KDF _kdf)
{
	h128 u = uuid(_address);
	store().recode(u, getPassword(u, _pass), _kdf);
}

bool KeyManager::load(std::string const& _pass)
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
			{
				m_keyInfo[m_addrLookup[(Address)i[0]] = (h128)i[1]] = KeyInfo((h256)i[2], (std::string)i[3]);
				cdebug << toString((Address)i[0]) << toString((h128)i[1]) << toString((h256)i[2]) << (std::string)i[3];
			}

			for (auto const& i: s[2])
				m_passwordInfo[(h256)i[0]] = (std::string)i[1];
			m_password = (string)s[3];
		}
		m_cachedPasswords[hashPassword(m_password)] = m_password;
		m_cachedPasswords[hashPassword(defaultPassword())] = defaultPassword();
		return true;
	}
	catch (...) {
		return false;
	}
}

Secret KeyManager::secret(Address const& _address, function<std::string()> const& _pass) const
{
	auto it = m_addrLookup.find(_address);
	if (it == m_addrLookup.end())
		return Secret();
	return secret(it->second, _pass);
}

Secret KeyManager::secret(h128 const& _uuid, function<std::string()> const& _pass) const
{
	return Secret(m_store.secret(_uuid, [&](){ return getPassword(_uuid, _pass); }));
}

std::string KeyManager::getPassword(h128 const& _uuid, function<std::string()> const& _pass) const
{
	auto kit = m_keyInfo.find(_uuid);
	if (kit != m_keyInfo.end())
	{
		auto it = m_cachedPasswords.find(kit->second.passHash);
		if (it != m_cachedPasswords.end())
			return it->second;
	}
	std::string p = _pass();
	m_cachedPasswords[hashPassword(p)] = p;
	return p;
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

h128 KeyManager::import(Secret const& _s, string const& _info, std::string const& _pass, string const& _passInfo)
{
	Address addr = KeyPair(_s).address();
	auto passHash = hashPassword(_pass);
	m_cachedPasswords[passHash] = _pass;
	m_passwordInfo[passHash] = _passInfo;
	auto uuid = m_store.importSecret(_s.asBytes(), _pass);
	m_keyInfo[uuid] = KeyInfo{passHash, _info};
	m_addrLookup[addr] = uuid;
	write(m_keysFile);
	return uuid;
}

void KeyManager::importExisting(h128 const& _uuid, std::string const& _info, std::string const& _pass, std::string const& _passInfo)
{
	bytes key = m_store.secret(_uuid, [&](){ return _pass; });
	if (key.empty())
		return;
	Address a = KeyPair(Secret(key)).address();
	auto passHash = hashPassword(_pass);
	if (!m_passwordInfo.count(passHash))
		m_passwordInfo[passHash] = _passInfo;
	if (!m_cachedPasswords.count(passHash))
		m_cachedPasswords[passHash] = _pass;
	m_addrLookup[a] = _uuid;
	m_keyInfo[_uuid].passHash = passHash;
	m_keyInfo[_uuid].info = _info;
	write(m_keysFile);
}

void KeyManager::kill(Address const& _a)
{
	auto id = m_addrLookup[_a];
	m_addrLookup.erase(_a);
	m_keyInfo.erase(id);
	m_store.kill(id);
}

AddressHash KeyManager::accounts() const
{
	AddressHash ret;
	for (auto const& i: m_addrLookup)
		if (m_keyInfo.count(i.second) > 0)
			ret.insert(i.first);
	return ret;
}

std::unordered_map<Address, std::pair<std::string, std::string>> KeyManager::accountDetails() const
{
	std::unordered_map<Address, std::pair<std::string, std::string>> ret;
	for (auto const& i: m_addrLookup)
		if (m_keyInfo.count(i.second) > 0)
			ret[i.first] = make_pair(m_keyInfo.at(i.second).info, m_passwordInfo.at(m_keyInfo.at(i.second).passHash));
	return ret;
}

h256 KeyManager::hashPassword(std::string const& _pass) const
{
	// TODO SECURITY: store this a bit more securely; Scrypt perhaps?
	return h256(pbkdf2(_pass, asBytes(m_password), 262144, 32));
}

bool KeyManager::write(std::string const& _keysFile) const
{
	if (!m_key)
		return false;
	write(m_key, _keysFile);
	return true;
}

void KeyManager::write(std::string const& _pass, std::string const& _keysFile) const
{
	bytes salt = h256::random().asBytes();
	writeFile(_keysFile + ".salt", salt);
	auto key = h128(pbkdf2(_pass, salt, 262144, 16));
	write(key, _keysFile);
}

void KeyManager::write(h128 const& _key, std::string const& _keysFile) const
{
	RLPStream s(4);
	s << 1;
	s.appendList(m_addrLookup.size());
	for (auto const& i: m_addrLookup)
		if (m_keyInfo.count(i.second))
		{
			auto ki = m_keyInfo.at(i.second);
			s.appendList(4) << i.first << i.second << ki.passHash << ki.info;
		}
	s.appendList(m_passwordInfo.size());
	for (auto const& i: m_passwordInfo)
		s.appendList(2) << i.first << i.second;
	s.append(m_password);

	writeFile(_keysFile, encryptSymNoAuth(_key, h128(), &s.out()));
	m_key = _key;
	m_cachedPasswords[hashPassword(defaultPassword())] = defaultPassword();

}
