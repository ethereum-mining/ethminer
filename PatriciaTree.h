#pragma once

#include <map>
#include "RLP.h"
#include "sha256.h"

namespace eth
{

using StringMap = std::map<std::string, std::string>;
using HexMap = std::map<bytes, std::string>;

u256 hash256(StringMap const& _s);
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

	u256 sha256() const;
	bytes rlp() const;

	void debugPrint();

	std::string const& at(std::string const& _key) const;
	void insert(std::string const& _key, std::string const& _value);
	void remove(std::string const& _key);

private:
	TrieNode* m_root;
};

}


