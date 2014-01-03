#include "Common.h"
#include "PatriciaTree.h"
using namespace std;
using namespace eth;

namespace eth
{

#define ENABLE_DEBUG_PRINT 0

#if ENABLE_DEBUG_PRINT
bool g_hashDebug = false;
#endif

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

std::string hexPrefixEncode(bytes const& _hexVector, bool _terminated, int _begin, int _end)
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

u256 hash256aux(HexMap const& _s, HexMap::const_iterator _begin, HexMap::const_iterator _end, unsigned _preLen)
{
#if ENABLE_DEBUG_PRINT
	static std::string s_indent;
	if (_preLen)
		s_indent += "  ";
#endif

	RLPStream rlp;
	if (_begin == _end)
		rlp << "";	// NULL
	else if (std::next(_begin) == _end)
	{
		// only one left - terminate with the pair.
		rlp << RLPList(2) << hexPrefixEncode(_begin->first, true, _preLen) << _begin->second;
#if ENABLE_DEBUG_PRINT
		if (g_hashDebug)
			std::cerr << s_indent << asHex(fConstBytes(_begin->first.data() + _preLen, _begin->first.size() - _preLen), 1) << ": " << _begin->second << " = " << sha256(rlp.out()) << std::endl;
#endif
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
#if ENABLE_DEBUG_PRINT
			if (g_hashDebug)
				std::cerr << s_indent << asHex(fConstBytes(_begin->first.data() + _preLen, sharedPre), 1) << ": " << std::endl;
#endif
			rlp << RLPList(2) << hexPrefixEncode(_begin->first, false, _preLen, sharedPre) << toBigEndianString(hash256aux(_s, _begin, _end, sharedPre));
#if ENABLE_DEBUG_PRINT
			if (g_hashDebug)
				std::cerr << s_indent << "= " << sha256(rlp.out()) << std::endl;
#endif
		}
		else
		{
			// otherwise enumerate all 16+1 entries.
			rlp << RLPList(17);
			auto b = _begin;
			if (_preLen == b->first.size())
			{
#if ENABLE_DEBUG_PRINT
				if (g_hashDebug)
					std::cerr << s_indent << "@: " << b->second << std::endl;
#endif
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
#if ENABLE_DEBUG_PRINT
					if (g_hashDebug)
						std::cerr << s_indent << std::hex << i << ": " << std::endl;
#endif
					rlp << toBigEndianString(hash256aux(_s, b, n, _preLen + 1));
				}
				b = n;
			}
			if (_preLen == _begin->first.size())
				rlp << _begin->second;
			else
				rlp << "";

#if ENABLE_DEBUG_PRINT
			if (g_hashDebug)
				std::cerr << s_indent << "= " << sha256(rlp.out()) << std::endl;
#endif
		}
	}
#if ENABLE_DEBUG_PRINT
	if (_preLen)
		s_indent.resize(s_indent.size() - 2);
#endif
	return sha256(rlp.out());
}

u256 hash256(StringMap const& _s)
{
	// build patricia tree.
	if (_s.empty())
		return sha256(RLPNull);
	HexMap hexMap;
	for (auto i = _s.rbegin(); i != _s.rend(); ++i)
		hexMap[toHex(i->first)] = i->second;
	return hash256aux(hexMap, hexMap.cbegin(), hexMap.cend(), 0);
}


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

	virtual std::string const& at(fConstBytes _key) const override;
	virtual TrieNode* insert(fConstBytes _key, std::string const& _value) override;
	virtual TrieNode* remove(fConstBytes _key) override;
	virtual bytes rlp() const override;

private:
	/// @returns (byte)-1 when no active branches, 16 when multiple active and the index of the active branch otherwise.
	byte activeBranch() const;

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

	virtual std::string const& at(fConstBytes _key) const override { return contains(_key) ? m_value : c_nullString; }
	virtual TrieNode* insert(fConstBytes _key, std::string const& _value) override;
	virtual TrieNode* remove(fConstBytes _key) override;
	virtual bytes rlp() const override { return rlpListBytes(hexPrefixEncode(m_ext, true), m_value); }

private:
	bool contains(fConstBytes _key) const { return _key.size() == m_ext.size() && !memcmp(_key.data(), m_ext.data(), _key.size()); }

	std::string m_value;
};

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

	virtual std::string const& at(fConstBytes _key) const override { assert(m_next); return contains(_key) ? m_next->at(_key.cropped(m_ext.size())) : c_nullString; }
	virtual TrieNode* insert(fConstBytes _key, std::string const& _value) override;
	virtual TrieNode* remove(fConstBytes _key) override;
	virtual bytes rlp() const override { assert(m_next); return rlpListBytes(hexPrefixEncode(m_ext, false), toBigEndianString(m_next->sha256())); }

private:
	bool contains(fConstBytes _key) const { return _key.size() >= m_ext.size() && !memcmp(_key.data(), m_ext.data(), m_ext.size()); }

	TrieNode* m_next;
};

TrieNode* TrieNode::newBranch(fConstBytes _k1, std::string const& _v1, fConstBytes _k2, std::string const& _v2)
{
	uint prefix = commonPrefix(_k1, _k2);

	TrieNode* ret;
	if (_k1.size() == prefix)
		ret = new TrieBranchNode(_k2[prefix], new TrieLeafNode(_k2.cropped(prefix + 1), _v2), _v1);
	else if (_k2.size() == prefix)
		ret = new TrieBranchNode(_k1[prefix], new TrieLeafNode(_k1.cropped(prefix + 1), _v1), _v2);
	else // both continue after split
		ret = new TrieBranchNode(_k1[prefix], new TrieLeafNode(_k1.cropped(prefix + 1), _v1), _k2[prefix], new TrieLeafNode(_k2.cropped(prefix + 1), _v2));

	if (prefix)
		// have shared prefix - split.
		ret = new TrieInfixNode(_k1.cropped(0, prefix), ret);

	return ret;
}

std::string const& TrieBranchNode::at(fConstBytes _key) const
{
	if (_key.empty())
		return m_value;
	else if (m_nodes[_key[0]] != nullptr)
		return m_nodes[_key[0]]->at(_key.cropped(1));
	return c_nullString;
}

TrieNode* TrieBranchNode::insert(fConstBytes _key, std::string const& _value)
{
	assert(_value.size());
	mark();
	if (_key.empty())
		m_value = _value;
	else
		if (!m_nodes[_key[0]])
			m_nodes[_key[0]] = new TrieLeafNode(_key.cropped(1), _value);
		else
			m_nodes[_key[0]] = m_nodes[_key[0]]->insert(_key.cropped(1), _value);
	return this;
}

TrieNode* TrieBranchNode::remove(fConstBytes _key)
{
	if (_key.empty())
		if (m_value.size())
		{
			m_value.clear();
			return rejig();
		}
		else {}
	else if (m_nodes[_key[0]] != nullptr)
	{
		m_nodes[_key[0]] = m_nodes[_key[0]]->remove(_key.cropped(1));
		return rejig();
	}
	return this;
}

TrieNode* TrieBranchNode::rejig()
{
	mark();
	byte n = activeBranch();

	if (n == (byte)-1 && m_value.size())
	{
		// switch to leaf
		auto r = new TrieLeafNode(fConstBytes(), m_value);
		delete this;
		return r;
	}
	else if (n < 16 && m_value.empty())
	{
		// only branching to n...
		if (auto b = dynamic_cast<TrieBranchNode*>(m_nodes[n]))
		{
			// switch to infix
			m_nodes[n] = nullptr;
			delete this;
			return new TrieInfixNode(fConstBytes(&n, 1), b);
		}
		else
		{
			auto x = dynamic_cast<TrieExtNode*>(m_nodes[n]);
			assert(x);
			// include in child
			pushFront(x->m_ext, n);
			m_nodes[n] = nullptr;
			delete this;
			return x;
		}
	}

	return this;
}

bytes TrieBranchNode::rlp() const
{
	RLPStream s;
	s << RLPList(17);
	for (auto i: m_nodes)
		s << (i ? toBigEndianString(i->sha256()) : "");
	s << m_value;
	return s.out();
}

byte TrieBranchNode::activeBranch() const
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

TrieNode* TrieInfixNode::insert(fConstBytes _key, std::string const& _value)
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

TrieNode* TrieInfixNode::remove(fConstBytes _key)
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

TrieNode* TrieLeafNode::insert(fConstBytes _key, std::string const& _value)
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

TrieNode* TrieLeafNode::remove(fConstBytes _key)
{
	if (contains(_key))
	{
		delete this;
		return nullptr;
	}
	return this;
}

Trie::~Trie()
{
	delete m_root;
}

u256 Trie::sha256() const
{
	return m_root ? m_root->sha256() : eth::sha256(RLPNull);
}

bytes Trie::rlp() const
{
	return m_root ? m_root->rlp() : RLPNull;
}

void Trie::debugPrint()
{
#if ENABLE_DEBUG_PRINT
	if (m_root)
		m_root->debugPrint();
#endif
}

std::string const& Trie::at(std::string const& _key) const
{
	if (!m_root)
		return c_nullString;
	auto h = toHex(_key);
	return m_root->at(fConstBytes(&h));
}

void Trie::insert(std::string const& _key, std::string const& _value)
{
	if (_value.empty())
		remove(_key);
	auto h = toHex(_key);
	m_root = m_root ? m_root->insert(&h, _value) : new TrieLeafNode(fConstBytes(&h), _value);
}

void Trie::remove(std::string const& _key)
{
	if (m_root)
	{
		auto h = toHex(_key);
		m_root = m_root->remove(&h);
	}
}

}
