#pragma once

#include <cassert>
#include <vector>
#include <string>

namespace eth
{

template <class _T>
class foreign
{
public:
	typedef _T value_type;
	typedef _T element_type;

	foreign(): m_data(nullptr), m_count(0) {}
	foreign(std::vector<typename std::remove_const<_T>::type>* _data): m_data(_data->data()), m_count(_data->size()) {}
	foreign(_T* _data, unsigned _count): m_data(_data), m_count(_count) {}

	explicit operator bool() const { return m_data && m_count; }

	std::vector<_T> toVector() const { return std::vector<_T>(m_data, m_data + m_count); }
	std::string toString() const { return std::string((char const*)m_data, ((char const*)m_data) + m_count); }
	template <class _T2> operator foreign<_T2>() const { assert(m_count * sizeof(_T) / sizeof(_T2) * sizeof(_T2) / sizeof(_T) == m_count); return foreign<_T2>((_T2*)m_data, m_count * sizeof(_T) / sizeof(_T2)); }

	_T* data() const { return m_data; }
	unsigned count() const { return m_count; }
	unsigned size() const { return m_count; }
	unsigned empty() const { return !m_count; }
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

}
