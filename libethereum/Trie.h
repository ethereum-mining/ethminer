/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Foobar is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file Trie.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <map>
#include <leveldb/db.h>
#include "RLP.h"
namespace ldb = leveldb;

namespace eth
{

bytes rlp256(StringMap const& _s);
h256 hash256(StringMap const& _s);
h256 hash256(u256Map const& _s);
std::string hexPrefixEncode(bytes const& _hexVector, bool _terminated = false, int _begin = 0, int _end = -1);

class TrieNode;

/**
 * @brief Merkle Patricia Tree "Trie": a modifed base-16 Radix tree.
 */
class Trie
{
public:
	Trie(): m_root(nullptr) {}
	~Trie();

	h256 hash256() const;
	bytes rlp() const;

	void debugPrint();

	std::string const& at(std::string const& _key) const;
	void insert(std::string const& _key, std::string const& _value);
	void remove(std::string const& _key);

private:
	TrieNode* m_root;
};

/*class HashDBFace
{
public:
	virtual void insert(h256 _key, bytesConstRef _value) = 0;
	virtual void remove(h256 _key) = 0;
	virtual std::string at(h256 _key) const = 0;
};

class HashDBOverlay
{
public:

	virtual void insert(h256 _key, bytesConstRef _value) = 0;
	virtual void remove(h256 _key) = 0;
	virtual std::string at(h256 _key) const = 0;
};*/

/**
 * @brief Merkle Patricia Tree "Trie": a modifed base-16 Radix tree.
 * This version uses an LDB backend - TODO: split off m_db & m_over into opaque key/value map layer and allow caching & testing without DB.
 * TODO: Implement!
 * TODO: Init function that inserts the SHA(emptyRLP) -> emptyRLP into the DB and sets m_root to SHA(emptyRLP).
 */
class GenericTrieDB
{
public:
	GenericTrieDB() {}
	GenericTrieDB(ldb::DB* _db, std::map<h256, std::string>* _overlay = nullptr): GenericTrieDB() { open(_db, c_null, _overlay); }
	GenericTrieDB(ldb::DB* _db, h256 _root, std::map<h256, std::string>* _overlay = nullptr): GenericTrieDB() { open(_db, _root, _overlay); }
	~GenericTrieDB() {}

	void open(ldb::DB* _db, h256 _root, std::map<h256, std::string>* _overlay = nullptr) { m_root = _root; m_db = _db; m_over = _overlay; }

	void setRoot(h256 _root) { m_root = _root; }
	h256 root() const { return m_root; }

	void debugPrint() {}

	std::string at(bytesConstRef _key) const { return std::string(); }
	void insert(bytesConstRef _key, bytesConstRef _value);
	void remove(bytesConstRef _key) {}

	// TODO: iterators.

private:
	void insertHelper(bytesConstRef _key, bytesConstRef _value, uint _begin, uint _end);

	std::string node(h256 _h) const { if (_h == c_null) return std::string(); if (m_over) { auto it = m_over->find(_h); if (it != m_over->end()) return it->second; } std::string ret; if (m_db) m_db->Get(m_readOptions, ldb::Slice((char const*)&m_root, 32), &ret); return ret; }
	void insertNode(h256 _h, bytesConstRef _v) const { m_over[_h] = _v; }
	h256 insertNode(bytesConstRef _v) const { auto h = sha3(_v); m_over[h] = _v; return h; }
	void killNode(h256 _h) const { m_over->erase(_h); }	// only from overlay - no killing from DB proper.

	static const h256 c_null;
	h256 m_root = c_null;

	ldb::DB* m_db = nullptr;
	std::map<h256, std::string>* m_over = nullptr;

	ldb::ReadOptions m_readOptions;
};

template <class KeyType>
class TrieDB: public GenericTrieDB
{
public:
	TrieDB() {}
	TrieDB(ldb::DB* _db, std::map<h256, std::string>* _overlay = nullptr): GenericTrieDB(_db, _overlay) {}
	TrieDB(ldb::DB* _db, h256 _root, std::map<h256, std::string>* _overlay = nullptr): GenericTrieDB() { open(_db, _root, _overlay); }

	std::string operator[](KeyType _k) const { return at(_k); }

	std::string at(KeyType _k) const { return GenericTrieDB::at(bytesConstRef((byte const*)&_k, sizeof(KeyType))); }
	void insert(KeyType _k, bytesConstRef _value) { GenericTrieDB::insert(bytesConstRef((byte const*)&_k, sizeof(KeyType)), _value); }
	void insert(KeyType _k, bytes const& _value) { insert(_k, bytesConstRef(&_value)); }
	void remove(KeyType _k) { GenericTrieDB::remove(bytesConstRef((byte const*)&_k, sizeof(KeyType))); }
};

}


