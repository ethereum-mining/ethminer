/* BEGIN COPYRIGHT
 *
 * This file is part of Noted.
 *
 * Copyright ©2011, 2012, Lancaster Logic Response Limited.
 *
 * Noted is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * Noted is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Noted.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <algorithm>
#include <type_traits>
#include <cmath>

#undef foreach
#define foreach BOOST_FOREACH

namespace lb
{

template <class T>
static T graphParameters(T _min, T _max, unsigned _divisions, T* o_from = 0, T* o_delta = 0, bool _forceMinor = false, T _divisor = 1)
{
	T uMin = _min / _divisor;
	T uMax = _max / _divisor;
	if (uMax == uMin || !_divisions)
	{
		if (o_delta && o_from)
		{
			*o_from = 0;
			*o_delta = 1;
		}
		return 1;
	}
	long double l10 = std::log10((uMax - uMin) / T(_divisions) * 5.5f);
	long double mt = std::pow(10.f, l10 - std::floor(l10));
	long double ep = std::pow(10.f, std::floor(l10));
	T inc = _forceMinor
			? ((mt > 6.f) ? ep / 2.f : (mt > 3.f) ? ep / 5.f : (mt > 1.2f) ? ep / 10.f : ep / 20.f)
			: ((mt > 6.f) ? ep * 2.f : (mt > 3.f) ? ep : (mt > 1.2f) ? ep / 2.f : ep / 5.f);
	if (inc == 0)
		inc = 1;
	if (o_delta && o_from)
	{
		(*o_from) = std::floor(uMin / inc) * inc * _divisor;
		(*o_delta) = (std::ceil(uMax / inc) - std::floor(uMin / inc)) * inc * _divisor;
	}
	else if (o_from)
	{
		(*o_from) = std::ceil(uMin / inc) * inc * _divisor;
	}
	return inc * _divisor;
}

struct GraphParametersForceMinor { GraphParametersForceMinor() {} };
static const GraphParametersForceMinor ForceMinor;

template <class T>
struct GraphParameters
{
	inline GraphParameters(std::pair<T, T> _range, unsigned _divisions)
	{
		incr = graphParameters(_range.first, _range.second, _divisions, &from, &delta, false);
		to = from + delta;
	}

	inline GraphParameters(std::pair<T, T> _range, unsigned _divisions, GraphParametersForceMinor)
	{
		incr = graphParameters(_range.first, _range.second, _divisions, &from, &delta, true);
		to = from + delta;
	}

	inline GraphParameters(std::pair<T, T> _range, unsigned _divisions, T _divisor)
	{
		major = graphParameters(_range.first, _range.second, _divisions, &from, &delta, false, _divisor);
		incr = graphParameters(_range.first, _range.second, _divisions, &from, &delta, true, _divisor);
		to = from + delta;
	}

	template <class S, class Enable = void>
	struct MajorDeterminor { static bool go(S _s, S _j) { S ip; S fp = std::modf(_s / _j + S(0.5), &ip); return fabs(fabs(fp) - 0.5) < 0.05; } };
	template <class S>
	struct MajorDeterminor<S, typename std::enable_if<std::is_integral<S>::value>::type> { static S go(S _s, S _j) { return _s % _j == 0; } };

	bool isMajor(T _t) const { return MajorDeterminor<T>::go(_t, major); }

	T from;
	T delta;
	T major;
	T to;
	T incr;
};

}
