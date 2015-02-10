#pragma once

#include <type_traits>
#include <cassert>
#include <vector>
#include <string>

namespace dev
{

template <class _T>
class vector_ref
{
public:
	using value_type = _T;
	using element_type = _T;
	using mutable_value_type = typename std::conditional<std::is_const<_T>::value, typename std::remove_const<_T>::type, _T>::type;

	vector_ref(): m_data(nullptr), m_count(0) {}
	vector_ref(_T* _data, size_t _count): m_data(_data), m_count(_count) {}
	vector_ref(std::string* _data): m_data((_T*)_data->data()), m_count(_data->size() / sizeof(_T)) {}
	vector_ref(typename std::conditional<std::is_const<_T>::value, std::vector<typename std::remove_const<_T>::type> const*, std::vector<_T>*>::type _data): m_data(_data->data()), m_count(_data->size()) {}
	vector_ref(typename std::conditional<std::is_const<_T>::value, std::string const&, std::string&>::type _data): m_data((_T*)_data.data()), m_count(_data.size() / sizeof(_T)) {}
#ifdef STORAGE_LEVELDB_INCLUDE_DB_H_
	vector_ref(leveldb::Slice const& _s): m_data(_s.data()), m_count(_s.size() / sizeof(_T)) {}
#endif
	explicit operator bool() const { return m_data && m_count; }

	bool contentsEqual(std::vector<mutable_value_type> const& _c) const { return _c.size() == m_count && !memcmp(_c.data(), m_data, m_count); }
	std::vector<mutable_value_type> toVector() const { return std::vector<mutable_value_type>(m_data, m_data + m_count); }
	std::vector<unsigned char> toBytes() const { return std::vector<unsigned char>((unsigned char const*)m_data, (unsigned char const*)m_data + m_count * sizeof(_T)); }
	std::string toString() const { return std::string((char const*)m_data, ((char const*)m_data) + m_count * sizeof(_T)); }
	template <class _T2> operator vector_ref<_T2>() const { assert(m_count * sizeof(_T) / sizeof(_T2) * sizeof(_T2) / sizeof(_T) == m_count); return vector_ref<_T2>((_T2*)m_data, m_count * sizeof(_T) / sizeof(_T2)); }

	_T* data() const { return m_data; }
	size_t count() const { return m_count; }
	size_t size() const { return m_count; }
	bool empty() const { return !m_count; }
	vector_ref<_T> next() const { return vector_ref<_T>(m_data + m_count, m_count); }
	vector_ref<_T> cropped(size_t _begin, size_t _count = ~size_t(0)) const { if (m_data && _begin + std::max(size_t(0), _count) <= m_count) return vector_ref<_T>(m_data + _begin, _count == ~size_t(0) ? m_count - _begin : _count); else return vector_ref<_T>(); }
	void retarget(_T const* _d, size_t _s) { m_data = _d; m_count = _s; }
	void retarget(std::vector<_T> const& _t) { m_data = _t.data(); m_count = _t.size(); }
	void copyTo(vector_ref<typename std::remove_const<_T>::type> _t) const { memcpy(_t.data(), m_data, std::min(_t.size(), m_count) * sizeof(_T)); }
	void populate(vector_ref<typename std::remove_const<_T>::type> _t) const { copyTo(_t); memset(_t.data() + m_count, 0, std::max(_t.size(), m_count) - m_count); }

	_T* begin() { return m_data; }
	_T* end() { return m_data + m_count; }
	_T const* begin() const { return m_data; }
	_T const* end() const { return m_data + m_count; }

	_T& operator[](size_t _i) { assert(m_data); assert(_i < m_count); return m_data[_i]; }
	_T const& operator[](size_t _i) const { assert(m_data); assert(_i < m_count); return m_data[_i]; }

	bool operator==(vector_ref<_T> const& _cmp) const { return m_data == _cmp.m_data && m_count == _cmp.m_count; }
	bool operator!=(vector_ref<_T> const& _cmp) const { return !operator==(_cmp); }

#ifdef STORAGE_LEVELDB_INCLUDE_DB_H_
	operator leveldb::Slice() const { return leveldb::Slice((char const*)m_data, m_count * sizeof(_T)); }
#endif

	void reset() { m_data = nullptr; m_count = 0; }

private:
	_T* m_data;
	size_t m_count;
};

template<class _T> vector_ref<_T const> ref(_T const& _t) { return vector_ref<_T const>(&_t, 1); }
template<class _T> vector_ref<_T> ref(_T& _t) { return vector_ref<_T>(&_t, 1); }
template<class _T> vector_ref<_T const> ref(std::vector<_T> const& _t) { return vector_ref<_T const>(&_t); }
template<class _T> vector_ref<_T> ref(std::vector<_T>& _t) { return vector_ref<_T>(&_t); }

}
