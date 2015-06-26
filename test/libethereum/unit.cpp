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
/** @file unit.cpp
 * @author Dimitry Khokhlov <Dimitry@ethdev.com>
 * @date 2015
 * libethereum unit test functions.
 */

#include <boost/filesystem/operations.hpp>
#include <boost/test/unit_test.hpp>
#include <libethereum/Defaults.h>
#include <libethereum/AccountDiff.h>

BOOST_AUTO_TEST_SUITE(libethereum)

BOOST_AUTO_TEST_CASE(AccountDiff)
{
	std::cout << "AccountDiff" << std::endl;
	dev::eth::AccountDiff accDiff;

	// exist = true	   exist_from = true		AccountChange::Deletion
	accDiff.exist = dev::Diff<bool>(true, false);
	BOOST_CHECK_MESSAGE(accDiff.changeType() == dev::eth::AccountChange::Deletion, "Account change type expected to be Deletion!");

	// exist = true	   exist_from = false		AccountChange::Creation
	accDiff.exist = dev::Diff<bool>(false, true);
	BOOST_CHECK_MESSAGE(accDiff.changeType() == dev::eth::AccountChange::Creation, "Account change type expected to be Creation!");

	// exist = false	   bn = true	sc = true	AccountChange::All
	accDiff.exist = dev::Diff<bool>(false, false);
	accDiff.nonce = dev::Diff<dev::u256>(1, 2);
	accDiff.code = dev::Diff<dev::bytes>(dev::fromHex("00"), dev::fromHex("01"));
	BOOST_CHECK_MESSAGE(accDiff.changeType() == dev::eth::AccountChange::All, "Account change type expected to be All!");

	// exist = false	   bn = true	sc = false  AccountChange::Intrinsic
	accDiff.exist = dev::Diff<bool>(false, false);
	accDiff.nonce = dev::Diff<dev::u256>(1, 2);
	accDiff.code = dev::Diff<dev::bytes>(dev::fromHex("00"), dev::fromHex("00"));
	BOOST_CHECK_MESSAGE(accDiff.changeType() == dev::eth::AccountChange::Intrinsic, "Account change type expected to be Intrinsic!");

	// exist = false	   bn = false   sc = true	AccountChange::CodeStorage
	accDiff.exist = dev::Diff<bool>(false, false);
	accDiff.nonce = dev::Diff<dev::u256>(1, 1);
	accDiff.balance = dev::Diff<dev::u256>(1, 1);
	accDiff.code = dev::Diff<dev::bytes>(dev::fromHex("00"), dev::fromHex("01"));
	BOOST_CHECK_MESSAGE(accDiff.changeType() == dev::eth::AccountChange::CodeStorage, "Account change type expected to be CodeStorage!");

	// exist = false	   bn = false   sc = false	AccountChange::None
	accDiff.exist = dev::Diff<bool>(false, false);
	accDiff.nonce = dev::Diff<dev::u256>(1, 1);
	accDiff.balance = dev::Diff<dev::u256>(1, 1);
	accDiff.code = dev::Diff<dev::bytes>(dev::fromHex("00"), dev::fromHex("00"));
	BOOST_CHECK_MESSAGE(accDiff.changeType() == dev::eth::AccountChange::None, "Account change type expected to be None!");
}

BOOST_AUTO_TEST_SUITE_END()
