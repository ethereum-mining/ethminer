#pragma once

#include <map>
#include "RLP.h"
#include "sha256.h"

#define ENABLE_DEBUG_PRINT 1

namespace eth
{

using StringMap = std::map<std::string, std::string>;
using HexMap = std::map<bytes, std::string>;

extern bool g_hashDebug;

/*
 * Hex-prefix Notation. First nibble has flags: oddness = 2^0 & termination = 2^1
 * [0,0,1,2,3,4,5]   0x10012345
 * [0,1,2,3,4,5]     0x00012345
 * [1,2,3,4,5]       0x112345
 * [0,0,1,2,3,4]     0x00001234
 * [0,1,2,3,4]       0x101234
 * [1,2,3,4]         0x001234
 * [0,0,1,2,3,4,5,T] 0x30012345
 * [0,0,1,2,3,4,T]   0x20001234
 * [0,1,2,3,4,5,T]   0x20012345
 * [1,2,3,4,5,T]     0x312345
 * [1,2,3,4,T]       0x201234
 */

inline std::string hexPrefixEncode(bytes const& _hexVector, bool _terminated = false, int _begin = 0, int _end = -1)
{
	uint begin = _begin;
	uint end = _end < 0 ? _hexVector.size() + 1 + _end : _end;
	bool termed = _terminated;
	bool odd = (end - begin) % 2;

	std::string ret(1, ((termed ? 2 : 0) | (odd ? 1 : 0)) * 16);
	if (odd)
	{
		ret[0] |= _hexVector[begin];
		++begin;
	}
	for (uint i = begin; i < end; i += 2)
		ret += _hexVector[i] * 16 + _hexVector[i + 1];
	return ret;
}

inline bytes toHex(std::string const& _s)
{
	std::vector<uint8_t> ret;
	ret.reserve(_s.size() * 2);
	for (auto i: _s)
	{
		ret.push_back(i / 16);
		ret.push_back(i % 16);
	}
	return ret;
}

inline std::string toBigEndianString(u256 _val)
{
	std::string ret;
	ret.resize(32);
	for (int i = 0; i <32; ++i, _val >>= 8)
		ret[31 - i] = (char)(uint8_t)_val;
	return ret;
}

inline u256 hash256aux(HexMap const& _s, HexMap::const_iterator _begin, HexMap::const_iterator _end, unsigned _preLen)
{
	static std::string s_indent;
	if (_preLen)
		s_indent += "  ";

	RLPStream rlp;
	if (_begin == _end)
	{
		rlp << "";	// NULL
	}
	else if (std::next(_begin) == _end)
	{
		// only one left - terminate with the pair.
		rlp << RLPList(2) << hexPrefixEncode(_begin->first, true, _preLen) << _begin->second;
		if (g_hashDebug)
			std::cerr << s_indent << asHex(fConstBytes(_begin->first.data() + _preLen, _begin->first.size() - _preLen), 1) << ": " << _begin->second << " = " << sha256(rlp.out()) << std::endl;
	}
	else
	{
		// find the number of common prefix nibbles shared
		// i.e. the minimum number of nibbles shared at the beginning between the first hex string and each successive.
		uint sharedPre = (uint)-1;
		uint c = 0;
		for (auto i = std::next(_begin); i != _end && sharedPre; ++i, ++c)
		{
			uint x = std::min(sharedPre, std::min(_begin->first.size(), i->first.size()));
			uint shared = _preLen;
			for (; shared < x && _begin->first[shared] == i->first[shared]; ++shared) {}
			sharedPre = std::min(shared, sharedPre);
		}
		if (sharedPre > _preLen)
		{
			// if they all have the same next nibble, we also want a pair.
			if (g_hashDebug)
				std::cerr << s_indent << asHex(fConstBytes(_begin->first.data() + _preLen, sharedPre), 1) << ": " << std::endl;
			rlp << RLPList(2) << hexPrefixEncode(_begin->first, false, _preLen, sharedPre) << toBigEndianString(hash256aux(_s, _begin, _end, sharedPre));
			if (g_hashDebug)
				std::cerr << s_indent << "= " << sha256(rlp.out()) << std::endl;
		}
		else
		{
			// otherwise enumerate all 16+1 entries.
			rlp << RLPList(17);
			auto b = _begin;
			if (_preLen == b->first.size())
			{
				if (g_hashDebug)
					std::cerr << s_indent << "@: " << b->second << std::endl;
				++b;
			}
			for (auto i = 0; i < 16; ++i)
			{
				auto n = b;
				for (; n != _end && n->first[_preLen] == i; ++n) {}
				if (b == n)
					rlp << "";
				else
				{
					if (g_hashDebug)
						std::cerr << s_indent << std::hex << i << ": " << std::endl;
					rlp << toBigEndianString(hash256aux(_s, b, n, _preLen + 1));
				}
				b = n;
			}
			if (_preLen == _begin->first.size())
				rlp << _begin->second;
			else
				rlp << "";

			if (g_hashDebug)
				std::cerr << s_indent << "= " << sha256(rlp.out()) << std::endl;
		}
	}
//	if (g_hashDebug)
//		std::cerr << std::hex << sha256(rlp.out()) << ": " << asHex(rlp.out()) << ": " << RLP(rlp.out()) << std::endl;
	if (_preLen)
		s_indent.resize(s_indent.size() - 2);
	return sha256(rlp.out());
}

inline u256 hash256(StringMap const& _s)
{
	// build patricia tree.
	if (_s.empty())
		return sha256(RLPNull);
	HexMap hexMap;
	for (auto i = _s.rbegin(); i != _s.rend(); ++i)
		hexMap[toHex(i->first)] = i->second;
//	for (auto const& i: _s)
//		hexMap[toHex(i.first)] = i.second;
	return hash256aux(hexMap, hexMap.cbegin(), hexMap.cend(), 0);
}

template <class _T, class _U> uint commonPrefix(_T const& _t, _U const& _u)
{
	uint s = std::min<uint>(_t.size(), _u.size());
	for (uint i = 0;; ++i)
		if (i == s || _t[i] != _u[i])
			return i;
	return s;
}

/**
 * @brief Merkle Patricia Tree: a modifed base-16 Radix tree.
 */
class TrieNode
{
public:
	TrieNode() {}
	virtual ~TrieNode() {}

	virtual std::string const& at(fConstBytes _key) const = 0;
	virtual TrieNode* insert(fConstBytes _key, std::string const& _value) = 0;
	virtual TrieNode* remove(fConstBytes _key) = 0;
	virtual bytes rlp() const = 0;

#if ENABLE_DEBUG_PRINT
	void debugPrint(std::string const& _indent = "") const { std::cerr << std::hex << sha256() << ":" << std::endl; debugPrintBody(_indent); }
#endif

	u256 sha256() const { /*if (!m_sha256)*/ m_sha256 = eth::sha256(rlp()); return m_sha256; }
	void mark() { m_sha256 = 0; }

protected:
#if ENABLE_DEBUG_PRINT
	virtual void debugPrintBody(std::string const& _indent = "") const = 0;
#endif

	static TrieNode* newBranch(fConstBytes _k1, std::string const& _v1, fConstBytes _k2, std::string const& _v2);

private:
	mutable u256 m_sha256 = 0;
};

static const std::string c_nullString;

class TrieExtNode: public TrieNode
{
public:
	TrieExtNode(fConstBytes _bytes): m_ext(_bytes.begin(), _bytes.end()) {}

	bytes m_ext;
};

class TrieBranchNode: public TrieNode
{
public:
	TrieBranchNode(std::string const& _value): m_value(_value)
	{
		memset(m_nodes.data(), 0, sizeof(TrieNode*) * 16);
	}

	TrieBranchNode(byte _i1, TrieNode* _n1, std::string const& _value = std::string()): m_value(_value)
	{
		memset(m_nodes.data(), 0, sizeof(TrieNode*) * 16);
		m_nodes[_i1] = _n1;
	}

	TrieBranchNode(byte _i1, TrieNode* _n1, byte _i2, TrieNode* _n2)
	{
		memset(m_nodes.data(), 0, sizeof(TrieNode*) * 16);
		m_nodes[_i1] = _n1;
		m_nodes[_i2] = _n2;
	}

	virtual ~TrieBranchNode()
	{
		for (auto i: m_nodes)
			delete i;
	}

#if ENABLE_DEBUG_PRINT
	virtual void debugPrintBody(std::string const& _indent) const
	{

		if (m_value.size())
			std::cerr << _indent << "@: " << m_value << std::endl;
		for (auto i = 0; i < 16; ++i)
			if (m_nodes[i])
			{
				std::cerr << _indent << std::hex << i << ": ";
				m_nodes[i]->debugPrint(_indent + "  ");
			}
	}
#endif

	virtual std::string const& at(fConstBytes _key) const override
	{
		if (_key.empty())
			return m_value;
		else if (m_nodes[_key[0]] != nullptr)
			return m_nodes[_key[0]]->at(_key.cropped(1));
		return c_nullString;
	}

	virtual TrieNode* insert(fConstBytes _key, std::string const& _value) override;

	virtual TrieNode* remove(fConstBytes _key) override;

	virtual bytes rlp() const override
	{
		RLPStream s;
		s << RLPList(17);
		for (auto i: m_nodes)
			s << (i ? toBigEndianString(i->sha256()) : "");
		s << m_value;
		return s.out();
	}

private:
	/// @returns (byte)-1 when no active branches, 16 when multiple active and the index of the active branch otherwise.
	byte activeBranch() const
	{
		byte n = (byte)-1;
		for (int i = 0; i < 16; ++i)
			if (m_nodes[i] != nullptr)
			{
				if (n == (byte)-1)
					n = i;
				else
					return 16;
			}
		return n;
	}

	TrieNode* rejig();

	std::array<TrieNode*, 16> m_nodes;
	std::string m_value;
};

class TrieLeafNode: public TrieExtNode
{
public:
	TrieLeafNode(fConstBytes _key, std::string const& _value): TrieExtNode(_key), m_value(_value) {}

#if ENABLE_DEBUG_PRINT
	virtual void debugPrintBody(std::string const& _indent) const
	{
		assert(m_value.size());
		std::cerr << _indent;
		if (m_ext.size())
			std::cerr << asHex(m_ext, 1) << ": ";
		else
			std::cerr << "@: ";
		std::cerr << m_value << std::endl;
	}
#endif

	virtual std::string const& at(fConstBytes _key) const override
	{
		return contains(_key) ? m_value : c_nullString;
	}

	virtual TrieNode* insert(fConstBytes _key, std::string const& _value) override
	{
		assert(_value.size());
		mark();
		if (contains(_key))
		{
			m_value = _value;
			return this;
		}
		else
		{
			// create new trie.
			auto n = TrieNode::newBranch(_key, _value, fConstBytes(&m_ext), m_value);
			delete this;
			return n;
		}
	}

	virtual TrieNode* remove(fConstBytes _key) override
	{
		if (contains(_key))
		{
			delete this;
			return nullptr;
		}
		return this;
	}

	virtual bytes rlp() const override
	{
		RLPStream s;
		s << RLPList(2) << hexPrefixEncode(m_ext, true) << m_value;
		return s.out();
	}

private:
	bool contains(fConstBytes _key) const { return _key.size() == m_ext.size() && !memcmp(_key.data(), m_ext.data(), _key.size()); }

	std::string m_value;
};

template <class _T> void trimFront(_T& _t, uint _elements)
{
	memmove(_t.data(), _t.data() + _elements, (_t.size() - _elements) * sizeof(_t[0]));
	_t.resize(_t.size() - _elements);
}

template <class _T, class _U> void pushFront(_T& _t, _U _e)
{
	_t.push_back(_e);
	memmove(_t.data() + 1, _t.data(), (_t.size() - 1) * sizeof(_e));
	_t[0] = _e;
}

class TrieInfixNode: public TrieExtNode
{
public:
	TrieInfixNode(fConstBytes _key, TrieNode* _next): TrieExtNode(_key), m_next(_next) {}
	virtual ~TrieInfixNode() { delete m_next; }

#if ENABLE_DEBUG_PRINT
	virtual void debugPrintBody(std::string const& _indent) const
	{
		std::cerr << _indent << asHex(m_ext, 1) << ": ";
		m_next->debugPrint(_indent + "  ");
	}
#endif

	virtual std::string const& at(fConstBytes _key) const override
	{
		assert(m_next);
		return contains(_key) ? m_next->at(_key.cropped(m_ext.size())) : c_nullString;
	}

	virtual TrieNode* insert(fConstBytes _key, std::string const& _value) override
	{
		assert(_value.size());
		mark();
		if (contains(_key))
		{
			m_next = m_next->insert(_key.cropped(m_ext.size()), _value);
			return this;
		}
		else
		{
			int prefix = commonPrefix(_key, m_ext);
			if (prefix)
			{
				// one infix becomes two infixes, then insert into the second
				// instead of pop_front()...
				trimFront(m_ext, prefix);

				return new TrieInfixNode(_key.cropped(0, prefix), insert(_key.cropped(prefix), _value));
			}
			else
			{
				// split here.
				auto f = m_ext[0];
				trimFront(m_ext, 1);
				TrieNode* n = m_ext.empty() ? m_next : this;
				if (n != this)
				{
					m_next = nullptr;
					delete this;
				}
				TrieBranchNode* ret = new TrieBranchNode(f, n);
				ret->insert(_key, _value);
				return ret;
			}
		}
	}

	virtual TrieNode* remove(fConstBytes _key) override
	{
		if (contains(_key))
		{
			mark();
			m_next = m_next->remove(_key.cropped(m_ext.size()));
			if (auto p = dynamic_cast<TrieExtNode*>(m_next))
			{
				// merge with child...
				m_ext.reserve(m_ext.size() + p->m_ext.size());
				for (auto i: p->m_ext)
					m_ext.push_back(i);
				p->m_ext = m_ext;
				p->mark();
				m_next = nullptr;
				delete this;
				return p;
			}
			if (!m_next)
			{
				delete this;
				return nullptr;
			}
		}
		return this;
	}

	virtual bytes rlp() const override
	{
		assert(m_next);
		RLPStream s;
		s << RLPList(2) << hexPrefixEncode(m_ext, false) << toBigEndianString(m_next->sha256());
		return s.out();
	}

private:
	bool contains(fConstBytes _key) const { return _key.size() >= m_ext.size() && !memcmp(_key.data(), m_ext.data(), m_ext.size()); }

	TrieNode* m_next;
};

class Trie
{
public:
	Trie(): m_root(nullptr) {}
	~Trie() { delete m_root; }

	u256 sha256() const { return m_root ? m_root->sha256() : eth::sha256(RLPNull); }
	bytes rlp() const { return m_root ? m_root->rlp() : RLPNull; }

	void debugPrint() { if (m_root) m_root->debugPrint(); }

	std::string const& at(std::string const& _key) const
	{
		if (!m_root)
			return c_nullString;
		auto h = toHex(_key);
		return m_root->at(fConstBytes(&h));
	}

	void insert(std::string const& _key, std::string const& _value)
	{
		if (_value.empty())
			remove(_key);
		auto h = toHex(_key);
		m_root = m_root ? m_root->insert(&h, _value) : new TrieLeafNode(fConstBytes(&h), _value);
	}

	void remove(std::string const& _key)
	{
		if (m_root)
		{
			auto h = toHex(_key);
			m_root = m_root->remove(&h);
		}
	}

private:
	TrieNode* m_root;
};

}


