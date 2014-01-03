#include "Common.h"
#include "PatriciaTree.h"
using namespace std;
using namespace eth;

bool eth::g_hashDebug = false;

/*
PatriciaTree::PatriciaTree(RLP const& _data)
{
	// Make tree based on _data
	assert(_data.isList());
	if (_data.isEmpty())
	{
		// NULL node.
	}
	else if (_data.isList() && _data.itemCount() == 2)
	{
		// Key-value pair
	}
	else if (_data.isList() && _data.itemCount() == 17)
	{
		// Branch
	}
}
*/

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
