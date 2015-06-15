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
/** @file SecretStore.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "SecretStore.h"
#include <thread>
#include <mutex>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <libdevcore/Log.h>
#include <libdevcore/Guards.h>
#include <libdevcore/SHA3.h>
#include <libdevcore/FileSystem.h>
#include <test/JsonSpiritHeaders.h>
using namespace std;
using namespace dev;
namespace js = json_spirit;
namespace fs = boost::filesystem;

static const int c_keyFileVersion = 3;

static js::mValue upgraded(std::string const& _s)
{
	js::mValue v;
	js::read_string(_s, v);
	if (v.type() != js::obj_type)
		return js::mValue();
	js::mObject ret = v.get_obj();
	unsigned version = ret.count("Version") ? stoi(ret["Version"].get_str()) : ret.count("version") ? ret["version"].get_int() : 0;
	if (version == 1)
	{
		// upgrade to version 2
		js::mObject old;
		swap(old, ret);

		ret["id"] = old["Id"];
		js::mObject c;
		c["ciphertext"] = old["Crypto"].get_obj()["CipherText"];
		c["cipher"] = "aes-128-cbc";
		{
			js::mObject cp;
			cp["iv"] = old["Crypto"].get_obj()["IV"];
			c["cipherparams"] = cp;
		}
		c["kdf"] = old["Crypto"].get_obj()["KeyHeader"].get_obj()["Kdf"];
		{
			js::mObject kp;
			kp["salt"] = old["Crypto"].get_obj()["Salt"];
			for (auto const& i: old["Crypto"].get_obj()["KeyHeader"].get_obj()["KdfParams"].get_obj())
				if (i.first != "SaltLen")
					kp[boost::to_lower_copy(i.first)] = i.second;
			c["kdfparams"] = kp;
		}
		c["sillymac"] = old["Crypto"].get_obj()["MAC"];
		c["sillymacjson"] = _s;
		ret["crypto"] = c;
		version = 2;
	}
	if (version == 2)
	{
		ret["crypto"].get_obj()["cipher"] = "aes-128-ctr";
		ret["crypto"].get_obj()["compat"] = "2";
		version = 3;
	}
	if (version == c_keyFileVersion)
		return ret;
	return js::mValue();
}

SecretStore::SecretStore(std::string const& _path): m_path(_path)
{
	load();
}

SecretStore::~SecretStore()
{
}

bytes SecretStore::secret(h128 const& _uuid, function<std::string()> const& _pass, bool _useCache) const
{
	(void)_pass;
	auto rit = m_cached.find(_uuid);
	if (_useCache && rit != m_cached.end())
		return rit->second;
	auto it = m_keys.find(_uuid);
	if (it == m_keys.end())
		return bytes();
	bytes key = decrypt(it->second.first, _pass());
	if (!key.empty())
		m_cached[_uuid] = key;
	return key;
}

h128 SecretStore::importSecret(bytes const& _s, std::string const& _pass)
{
	h128 r = h128::random();
	m_cached[r] = _s;
	m_keys[r] = make_pair(encrypt(_s, _pass), std::string());
	save();
	return r;
}

void SecretStore::kill(h128 const& _uuid)
{
	m_cached.erase(_uuid);
	if (m_keys.count(_uuid))
	{
		boost::filesystem::remove(m_keys[_uuid].second);
		m_keys.erase(_uuid);
	}
}

void SecretStore::clearCache() const
{
	m_cached.clear();
}

void SecretStore::save(std::string const& _keysPath)
{
	fs::path p(_keysPath);
	boost::filesystem::create_directories(p);
	for (auto& k: m_keys)
	{
		std::string uuid = toUUID(k.first);
		std::string filename = (p / uuid).string() + ".json";
		js::mObject v;
		js::mValue crypto;
		js::read_string(k.second.first, crypto);
		v["crypto"] = crypto;
		v["id"] = uuid;
		v["version"] = c_keyFileVersion;
		writeFile(filename, js::write_string(js::mValue(v), true));
		if (!k.second.second.empty() && k.second.second != filename)
			boost::filesystem::remove(k.second.second);
		k.second.second = filename;
	}
}

void SecretStore::load(std::string const& _keysPath)
{
	fs::path p(_keysPath);
	boost::filesystem::create_directories(p);
	for (fs::directory_iterator it(p); it != fs::directory_iterator(); ++it)
		if (is_regular_file(it->path()))
			readKey(it->path().string(), true);
}

h128 SecretStore::readKey(std::string const& _file, bool _deleteFile)
{
	cnote << "Reading" << _file;
	return readKeyContent(contentsString(_file), _deleteFile ? _file : string());
}

h128 SecretStore::readKeyContent(std::string const& _content, std::string const& _file)
{
	js::mValue u = upgraded(_content);
	if (u.type() == js::obj_type)
	{
		js::mObject& o = u.get_obj();
		auto uuid = fromUUID(o["id"].get_str());
		m_keys[uuid] = make_pair(js::write_string(o["crypto"], false), _file);
		return uuid;
	}
	else
		cwarn << "Invalid JSON in key file" << _file;
	return h128();
}

bool SecretStore::recode(h128 const& _uuid, string const& _newPass, std::function<std::string()> const& _pass, KDF _kdf)
{
//	cdebug << "recode:" << toUUID(_uuid);
	bytes s = secret(_uuid, _pass, true);
	if (s.empty())
		return false;
	m_keys[_uuid].first = encrypt(s, _newPass, _kdf);
	save();
	return true;
}

std::string SecretStore::encrypt(bytes const& _v, std::string const& _pass, KDF _kdf)
{
	js::mObject ret;

	// KDF info
	unsigned dklen = 32;
	bytes salt = h256::random().asBytes();
	bytes derivedKey;
	if (_kdf == KDF::Scrypt)
	{
		unsigned iterations = 262144;
		unsigned p = 1;
		unsigned r = 8;
		ret["kdf"] = "scrypt";
		{
			js::mObject params;
			params["n"] = (int64_t)iterations;
			params["r"] = (int)r;
			params["p"] = (int)p;
			params["dklen"] = (int)dklen;
			params["salt"] = toHex(salt);
			ret["kdfparams"] = params;
		}
		derivedKey = scrypt(_pass, salt, iterations, r, p, dklen);
	}
	else
	{
		unsigned iterations = 262144;
		ret["kdf"] = "pbkdf2";
		{
			js::mObject params;
			params["prf"] = "hmac-sha256";
			params["c"] = (int)iterations;
			params["salt"] = toHex(salt);
			params["dklen"] = (int)dklen;
			ret["kdfparams"] = params;
		}
		derivedKey = pbkdf2(_pass, salt, iterations, dklen);
	}
//	cdebug << "derivedKey" << toHex(derivedKey);

	// cipher info
	ret["cipher"] = "aes-128-ctr";
	h128 key(derivedKey, h128::AlignLeft);
//	cdebug << "cipherKey" << key.hex();
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
	h256 mac = sha3(ref(derivedKey).cropped(16, 16).toBytes() + cipherText);
//	cdebug << "macBody" << toHex(ref(derivedKey).cropped(16, 16).toBytes() + cipherText);
//	cdebug << "mac" << mac.hex();
	ret["mac"] = toHex(mac.ref());

	return js::write_string((js::mValue)ret, true);
}

bytes SecretStore::decrypt(std::string const& _v, std::string const& _pass)
{
	js::mObject o;
	{
		js::mValue ov;
		js::read_string(_v, ov);
		o = ov.get_obj();
	}

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
	else if (o["kdf"].get_str() == "scrypt")
	{
		auto p = o["kdfparams"].get_obj();
		derivedKey = scrypt(_pass, fromHex(p["salt"].get_str()), p["n"].get_int(), p["r"].get_int(), p["p"].get_int(), p["dklen"].get_int());
	}
	else
	{
		cwarn << "Unknown KDF" << o["kdf"].get_str() << "not supported.";
		return bytes();
	}

	if (derivedKey.size() < 32 && !(o.count("compat") && o["compat"].get_str() == "2"))
	{
		cwarn << "Derived key's length too short (<32 bytes)";
		return bytes();
	}

	bytes cipherText = fromHex(o["ciphertext"].get_str());

	// check MAC
	if (o.count("mac"))
	{
		h256 mac(o["mac"].get_str());
		h256 macExp;
		if (o.count("compat") && o["compat"].get_str() == "2")
			macExp = sha3(bytesConstRef(&derivedKey).cropped(derivedKey.size() - 16).toBytes() + cipherText);
		else
			macExp = sha3(bytesConstRef(&derivedKey).cropped(16, 16).toBytes() + cipherText);
		if (mac != macExp)
		{
			cwarn << "Invalid key - MAC mismatch; expected" << toString(macExp) << ", got" << toString(mac);
			return bytes();
		}
	}
	else if (o.count("sillymac"))
	{
		h256 mac(o["sillymac"].get_str());
		h256 macExp = sha3(asBytes(o["sillymacjson"].get_str()) + bytesConstRef(&derivedKey).cropped(derivedKey.size() - 16).toBytes() + cipherText);
		if (mac != macExp)
		{
			cwarn << "Invalid key - MAC mismatch; expected" << toString(macExp) << ", got" << toString(mac);
			return bytes();
		}
	}
	else
		cwarn << "No MAC. Proceeding anyway.";

	// decrypt
	if (o["cipher"].get_str() == "aes-128-ctr")
	{
		auto params = o["cipherparams"].get_obj();
		h128 iv(params["iv"].get_str());
		if (o.count("compat") && o["compat"].get_str() == "2")
		{
			h128 key(sha3(h128(derivedKey, h128::AlignRight)), h128::AlignRight);
			return decryptSymNoAuth(key, iv, &cipherText);
		}
		else
			return decryptSymNoAuth(h128(derivedKey, h128::AlignLeft), iv, &cipherText);
	}
	else
	{
		cwarn << "Unknown cipher" << o["cipher"].get_str() << "not supported.";
		return bytes();
	}
}
