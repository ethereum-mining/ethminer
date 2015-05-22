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
/** @file KeyManager.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <functional>
#include <mutex>
#include <libdevcrypto/SecretStore.h>
#include <libdevcore/FileSystem.h>

namespace dev
{
namespace eth
{
class UnknownPassword: public Exception {};

struct KeyInfo
{
	KeyInfo() = default;
	KeyInfo(h256 const& _passHash, std::string const& _info): passHash(_passHash), info(_info) {}
	h256 passHash;
	std::string info;
};

static const auto DontKnowThrow = [](){ throw UnknownPassword(); return std::string(); };

enum class SemanticPassword
{
	Existing,
	Master
};

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
	KeyManager(std::string const& _keysFile = defaultPath(), std::string const& _secretsPath = SecretStore::defaultPath());
	~KeyManager();

	void setKeysFile(std::string const& _keysFile) { m_keysFile = _keysFile; }
	std::string const& keysFile() const { return m_keysFile; }

	bool exists() const;
	void create(std::string const& _pass);
	bool load(std::string const& _pass);
	void save(std::string const& _pass) const { write(_pass, m_keysFile); }

	void notePassword(std::string const& _pass) { m_cachedPasswords[hashPassword(_pass)] = _pass; }
	void noteHint(std::string const& _pass, std::string const& _hint) { if (!_hint.empty()) m_passwordInfo[hashPassword(_pass)] = _hint; }
	bool haveHint(std::string const& _pass) const { auto h = hashPassword(_pass); return m_cachedPasswords.count(h) && !m_cachedPasswords.at(h).empty(); }

	AddressHash accounts() const;
	std::unordered_map<Address, std::pair<std::string, std::string>> accountDetails() const;
	std::string const& hint(Address const& _a) const { try { return m_passwordInfo.at(m_keyInfo.at(m_addrLookup.at(_a)).passHash); } catch (...) { return EmptyString; } }

	h128 uuid(Address const& _a) const;
	Address address(h128 const& _uuid) const;

	h128 import(Secret const& _s, std::string const& _info, std::string const& _pass, std::string const& _passInfo);
	h128 import(Secret const& _s, std::string const& _info) { return import(_s, _info, defaultPassword(), std::string()); }

	SecretStore& store() { return m_store; }
	void importExisting(h128 const& _uuid, std::string const& _info, std::string const& _pass, std::string const& _passInfo);
	void importExisting(h128 const& _uuid, std::string const& _info) { importExisting(_uuid, _info, defaultPassword(), std::string()); }

	Secret secret(Address const& _address, std::function<std::string()> const& _pass = DontKnowThrow) const;
	Secret secret(h128 const& _uuid, std::function<std::string()> const& _pass = DontKnowThrow) const;

	bool recode(Address const& _address, SemanticPassword _newPass, std::function<std::string()> const& _pass = DontKnowThrow, KDF _kdf = KDF::Scrypt);
	bool recode(Address const& _address, std::string const& _newPass, std::string const& _hint, std::function<std::string()> const& _pass = DontKnowThrow, KDF _kdf = KDF::Scrypt);

	void kill(h128 const& _id) { kill(address(_id)); }
	void kill(Address const& _a);

	static std::string defaultPath() { return getDataDir("ethereum") + "/keys.info"; }

private:
	std::string getPassword(h128 const& _uuid, std::function<std::string()> const& _pass = DontKnowThrow) const;
	std::string getPassword(h256 const& _passHash, std::function<std::string()> const& _pass = DontKnowThrow) const;
	std::string defaultPassword(std::function<std::string()> const& _pass = DontKnowThrow) const { return getPassword(m_master, _pass); }
	h256 hashPassword(std::string const& _pass) const;

	// Only use if previously loaded ok.
	// @returns false if wasn't previously loaded ok.
	bool write() const { return write(m_keysFile); }
	bool write(std::string const& _keysFile) const;
	void write(std::string const& _pass, std::string const& _keysFile) const;
	void write(h128 const& _key, std::string const& _keysFile) const;

	// Ethereum keys.
	std::unordered_map<Address, h128> m_addrLookup;
	std::unordered_map<h128, KeyInfo> m_keyInfo;
	std::unordered_map<h256, std::string> m_passwordInfo;

	// Passwords that we're storing.
	mutable std::unordered_map<h256, std::string> m_cachedPasswords;

	// DEPRECATED.
	// Used to be the default password for keys in the keystore, stored in the keys file.
	// Now the default password is based off the key of the keys file directly, so this is redundant
	// except for the fact that people have existing keys stored with it. Leave for now until/unless
	// we have an upgrade strategy.
	std::string m_password;

	mutable std::string m_keysFile;
	mutable h128 m_key;
	mutable h256 m_master;
	SecretStore m_store;
};

}
}
