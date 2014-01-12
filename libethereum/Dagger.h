#pragma once

#include "Common.h"

namespace eth
{

/// Functions are not re-entrant. If you want to multi-thread, then use different classes for each thread.
class Dagger
{
public:
	Dagger(h256 _hash);
	~Dagger();
	
	u256 node(uint_fast32_t _L, uint_fast32_t _i) const;
	u256 eval(u256 _N);
	u256 search(uint _msTimeout, u256 _diff);

	static u256 bound(u256 _diff);

private:
	u256 m_hash;
	u256 m_xn;
};

}


