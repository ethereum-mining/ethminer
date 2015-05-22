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
/** @file SecretStore.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <functional>
#include <mutex>
#include <libdevcore/FixedHash.h>
#include <libdevcore/FileSystem.h>
#include "Common.h"

namespace dev
{

enum class KDF {
	PBKDF2_SHA256,
	Scrypt,
};

class SecretStore
{
public:
	SecretStore(std::string const& _path = defaultPath());
	~SecretStore();

	bytes secret(h128 const& _uuid, std::function<std::string()> const& _pass, bool _useCache = true) const;
	h128 importKey(std::string const& _file) { auto ret = readKey(_file, false); if (ret) save(); return ret; }
	h128 importSecret(bytes const& _s, std::string const& _pass);
	bool recode(h128 const& _uuid, std::string const& _newPass, std::function<std::string()> const& _pass, KDF _kdf = KDF::Scrypt);
	void kill(h128 const& _uuid);

	std::vector<h128> keys() const { return keysOf(m_keys); }

	// Clear any cached keys.
	void clearCache() const;

	static std::string defaultPath() { return getDataDir("web3") + "/keys"; }

private:
	void save(std::string const& _keysPath);
	void load(std::string const& _keysPath);
	void save() { save(m_path); }
	void load() { load(m_path); }
	static std::string encrypt(bytes const& _v, std::string const& _pass, KDF _kdf = KDF::Scrypt);
	static bytes decrypt(std::string const& _v, std::string const& _pass);
	h128 readKey(std::string const& _file, bool _deleteFile);

	mutable std::unordered_map<h128, bytes> m_cached;
	std::unordered_map<h128, std::pair<std::string, std::string>> m_keys;

	std::string m_path;
};

}

