/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file stdafx.h
 * @author Tim Hughes <tim@twistedfury.com>
 * @date 2014
 */

#pragma once

#include <string>
#include <array>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <ctime>
#include <chrono>
#include <cassert>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <type_traits>
#include <mutex>
#include <atomic>
#include <random>
#include <exception>
#include <memory>
#include <algorithm>

#include <assert.h>

#include <boost/filesystem.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/asio/detail/socket_types.hpp>
#include <boost/thread.hpp>

#pragma warning(push)
#pragma warning(disable: 4100 4267)
#include <leveldb/db.h>
#pragma warning(pop)


