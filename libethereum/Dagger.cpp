#include <boost/detail/endian.hpp>
#include <chrono>
#include <array>
#pragma warning(push)
#pragma warning(disable:4244)
#include <sha3.h>
#pragma warning(pop)
#include <random>
#include "Common.h"
#include "Dagger.h"
using namespace std;
using namespace std::chrono;

namespace eth
{

Dagger::Dagger(h256 _hash): m_hash(_hash)
{
}

Dagger::~Dagger()
{
}

u256 Dagger::bound(u256 _diff)
{
	return (u256)((bigint(1) << 256) / _diff);
}

u256 Dagger::search(uint _msTimeout, u256 _diff)
{
	static mt19937_64 s_engine((std::random_device())());
	u256 b = bound(_diff);

	auto start = steady_clock::now();

	while (steady_clock::now() - start < milliseconds(_msTimeout))
		for (uint sp = std::uniform_int_distribution<uint>()(s_engine), j = 0; j < 1000; ++j, ++sp)
			if (eval(sp) < b)
				return sp;
	return 0;
}

template <class _T, class _U>
inline void update(_T& _sha, _U const& _value)
{
	int i = 0;
	for (_U v = _value; v; ++i, v >>= 8) {}
	byte buf[32];
	bytesRef bufRef(buf, i);
	toBigEndian(_value, bufRef);
	_sha.Update(buf, i);
}

template <class _T>
inline u256 get(_T& _sha)
{
	byte buf[32];
	_sha.TruncatedFinal(buf, 32);
	return fromBigEndian<u256>(bytesConstRef(buf, 32));
}

u256 Dagger::node(uint_fast32_t _L, uint_fast32_t _i) const
{
	if (_L == _i)
		return m_hash;
	u256 m = (_L == 9) ? 16 : 3;
	CryptoPP::SHA3_256 bsha;
	for (uint_fast32_t k = 0; k < m; ++k)
	{
		CryptoPP::SHA3_256 sha;
		update(sha, m_hash);
		update(sha, m_xn);
		update(sha, (u256)_L);
		update(sha, (u256)_i);
		update(sha, (u256)k);
		uint_fast32_t pk = (uint_fast32_t)get(sha) & ((1 << ((_L - 1) * 3)) - 1);
		auto u = node(_L - 1, pk);
		update(bsha, u);
	}
	return get(bsha);
}

u256 Dagger::eval(u256 _N)
{
	m_xn = _N >> 26;				// with xn = floor(n / 2^26) -> assuming this is with xn = floor(N / 2^26)
	CryptoPP::SHA3_256 bsha;
	for (uint_fast32_t k = 0; k < 4; ++k)
	{
		//sha256(D || xn || i || k)		-> sha256(D || xn || k)	- there's no 'i' here!
		CryptoPP::SHA3_256 sha;
		update(sha, m_hash);
		update(sha, m_xn);
		update(sha, _N);
		update(sha, (u256)k);
		uint_fast32_t pk = (uint_fast32_t)get(sha) & 0x1ffffff;	// mod 8^8 * 2  [ == mod 2^25 ?! ] [ == & ((1 << 25) - 1) ] [ == & 0x1ffffff ]
		auto u = node(9, pk);
		update(bsha, u);
	}
	return get(bsha);
}

}
