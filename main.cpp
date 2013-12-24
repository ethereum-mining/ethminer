#include <iostream>
#include <cstdint>
#include <cassert>
#include <type_traits>
#include <vector>
#include <string>
#include <memory>

typedef uint8_t byte;

template <class _T>
class foreign
{
public:
	typedef _T value_type;
	typedef _T element_type;

	foreign(): m_data(nullptr), m_count(0) {}
	foreign(std::vector<typename std::remove_const<_T>::type>* _data): m_data(_data->data()), m_count(_data->size()) {}
	foreign(_T* _data, unsigned _count): m_data(_data), m_count(_count) {}
	foreign(std::shared_ptr<std::vector<typename std::remove_const<_T>::type> > const& _data): m_data(_data->data()), m_count(_data->size()) {}

	explicit operator bool() const { return m_data && m_count; }

	std::vector<_T> toVector() const { return std::vector<_T>(m_data, m_data + m_count); }
	std::string toString() const { return std::string((char const*)m_data, ((char const*)m_data) + m_count); }
	template <class _T2> operator foreign<_T2>() const { assert(m_count * sizeof(_T) / sizeof(_T2) * sizeof(_T2) / sizeof(_T) == m_count); return foreign<_T2>((_T2*)m_data, m_count * sizeof(_T) / sizeof(_T2)); }

	_T* data() const { return m_data; }
	unsigned count() const { return m_count; }
	unsigned size() const { return m_count; }
	foreign<_T> next() const { return foreign<_T>(m_data + m_count, m_count); }
	foreign<_T> cropped(unsigned _begin, int _count = -1) const { if (m_data && _begin + std::max(0, _count) <= m_count) return foreign<_T>(m_data + _begin, _count < 0 ? m_count - _begin : _count); else return foreign<_T>(); }
	void retarget(_T const* _d, size_t _s) { m_data = _d; m_count = _s; }
	void retarget(std::vector<_T> const& _t) { m_data = _t.data(); m_count = _t.size(); }

	_T* begin() { return m_data; }
	_T* end() { return m_data + m_count; }
	_T const* begin() const { return m_data; }
	_T const* end() const { return m_data + m_count; }

	_T& operator[](unsigned _i) { assert(m_data); assert(_i < m_count); return m_data[_i]; }
	_T const& operator[](unsigned _i) const { assert(m_data); assert(_i < m_count); return m_data[_i]; }

	void reset() { m_data = nullptr; m_count = 0; }

private:
	_T* m_data;
	unsigned m_count;
};

typedef foreign<byte> Bytes;
class RLP;
typedef std::vector<RLP> RLPs;

class RLP
{
public:
	RLP() {}
	RLP(Bytes _d): m_data(_d) {}

	explicit operator bool() const { return !isNull(); }
	bool isNull() const { return m_data.size() == 0; }
	bool isString() const { assert(!isNull()); return m_data[0] >= 0x40 && m_data[0] < 0x80; }
	bool isList() const { assert(!isNull()); return m_data[0] >= 0x80 && m_data[0] < 0xc0; }
	bool isInt() const { assert(!isNull()); return m_data[0] < 0x40; }
	bool isSmallString() const { assert(!isNull()); return m_data[0] >= 0x40 && m_data[0] < 0x78; }
	bool isSmallList() const { assert(!isNull()); return m_data[0] >= 0x80 && m_data[0] < 0xb8; }
	bool isSmallInt() const { assert(!isNull()); return m_data[0] < 0x20; }
	int intSize() const { return (!isInt() || isSmallInt()) ? 0 : m_data[0] - 0x1f; }

	uint64_t size() const
	{
		if (isInt())
			return 1 + intSize();
		if (isString())
			return payload().data() - m_data.data() + items();
		if (isList())
		{
			Bytes d = payload();
			uint64_t c = items();
			for (uint64_t i = 0; i < c; ++i, d = d.cropped(RLP(d).size())) {}
			return d.data() - m_data.data();
		}
		return 0;
	}

	std::string toString() const
	{
		if (!isString())
			return std::string();
		return payload().cropped(0, items()).toString();
	}

	uint64_t toInt(uint64_t _def = 0) const
	{
		if (!isInt())
			return _def;
		if (isSmallInt())
			return m_data[0];
		uint64_t ret = 0;
		for (int i = 0; i < intSize(); ++i)
			ret = (ret << 8) | m_data[i + 1];
		return ret;
	}

	RLPs toList() const
	{
		RLPs ret;
		if (!isList())
			return ret;
		uint64_t c = items();
		Bytes d = payload();
		for (uint64_t i = 0; i < c; ++i, d = d.cropped(RLP(d).size()))
			ret.push_back(RLP(d));
		return ret;
	}

/*	static uint64_t intLength(Bytes _b)
	{
		return _b[0] < 253 ? 1 : _b[0] == 253 ? 3 : _b[0] == 254 ? 5 : 9;
	}

	static uint64_t toInt(Bytes _b)
	{
		return _b[0] < 253 ?
					_b[0] :
				_b[0] == 253 ?
					(((uint64_t)_b[1]) << 8) |
					_b[2] :
				_b[0] == 254 ?
					(((uint64_t)_b[1]) << 24) |
					(((uint64_t)_b[2]) << 16) |
					(((uint64_t)_b[3]) << 8) |
					_b[4]
				: (
					(((uint64_t)_b[1]) << 56) |
					(((uint64_t)_b[2]) << 48) |
					(((uint64_t)_b[3]) << 40) |
					(((uint64_t)_b[4]) << 32) |
					(((uint64_t)_b[5]) << 24) |
					(((uint64_t)_b[6]) << 16) |
					(((uint64_t)_b[7]) << 8) |
					_b[8]
				);
	}*/

private:
	uint64_t items() const
	{
		assert(isString() || isList());
		auto n = (m_data[0] & 0x3f);
		if (n < 0x38)
			return n;
		uint64_t ret = 0;
		for (int i = 0; i < n; ++i)
			ret = (ret << 8) | m_data[i + 1];
		return ret;
	}

	Bytes payload() const
	{
		assert(isString() || isList());
		auto n = (m_data[0] & 0x3f);
		return m_data.cropped(1 + (n < 0x38 ? 0 : n));
	}

	Bytes m_data;
};

std::ostream& operator<<(std::ostream& _out, RLP _d)
{
	if (_d.isNull())
		_out << "null";
	else if (_d.isInt())
		_out << _d.toInt();
	else if (_d.isString())
		_out << "\"" << _d.toString() << "\"";
	else if (_d.isList())
	{
		_out << "[";
		int j = 0;
		for (auto i: _d.toList())
			_out << (j++ ? ", " : " ") << i;
		_out << "]";
	}

	return _out;
}

using namespace std;

int main()
{
	cout << "Hello World!" << endl;
	return 0;
}

