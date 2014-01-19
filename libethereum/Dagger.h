#pragma once

#include "Common.h"

#define FAKE_DAGGER 1

namespace eth
{

#if FAKE_DAGGER

class Dagger
{
public:
	static h256 eval(h256 const& _root, u256 const& _nonce) { h256 b = (h256)((u256)_root ^ _nonce); return sha3(bytesConstRef((byte const*)&b, 32)); }
	static bool verify(h256 const& _root, u256 const& _nonce, u256 const& _difficulty) { return (u256)eval(_root, _nonce) > _difficulty; }

	bool mine(u256& o_solution, h256 const& _root, u256 const& _difficulty, uint _msTimeout = 100);
};

#else

/// Functions are not re-entrant. If you want to multi-thread, then use different classes for each thread.
class Dagger
{
public:
	Dagger();
	~Dagger();
	
	static u256 bound(u256 const& _difficulty);
	static h256 eval(h256 const& _root, u256 const& _nonce);
	static bool verify(h256 const& _root, u256 const& _nonce, u256 const& _difficulty);

	bool mine(u256& o_solution, h256 const& _root, u256 const& _difficulty, uint _msTimeout = 100);

private:

	static h256 node(h256 const& _root, h256 const& _xn, uint_fast32_t _L, uint_fast32_t _i);

	h256 m_root;
	u256 m_nonce;
};

#endif

}


