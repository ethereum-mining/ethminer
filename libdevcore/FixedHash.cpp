/*
    This file is part of ethminer.

    ethminer is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ethminer is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ethminer.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file FixedHash.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <boost/algorithm/string.hpp>

#include "FixedHash.h"

using namespace std;
using namespace dev;

std::random_device dev::s_fixedHashEngine;
