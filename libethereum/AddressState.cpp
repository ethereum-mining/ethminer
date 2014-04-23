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
/** @file AddressState.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "AddressState.h"
#include "CommonEth.h"
using namespace std;
using namespace eth;

AddressState::AddressState(u256 _balance, u256 _nonce, bytesConstRef _code):
	m_isAlive(true),
	m_isComplete(true),
	m_balance(_balance),
	m_nonce(_nonce),
	m_code(_code.toBytes())
{}
