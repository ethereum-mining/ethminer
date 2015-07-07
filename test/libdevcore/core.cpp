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
/** @file core.cpp
 * @author Dimitry Khokhlov <winsvega@mail.ru>
 * @date 2014
 * CORE test functions.
 */

#include <boost/test/unit_test.hpp>
#include <libdevcore/CommonIO.h>
#include <test/TestHelper.h>

BOOST_AUTO_TEST_SUITE(CoreLibTests)

BOOST_AUTO_TEST_CASE(byteRef)
{	
	cnote << "bytesRef copyTo and toString...";
	dev::bytes originalSequence = dev::fromHex("0102030405060708091011121314151617181920212223242526272829303132");
	dev::bytesRef out(&originalSequence.at(0), 32);
	dev::h256 hash32("1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347");
	hash32.ref().copyTo(out);

	BOOST_CHECK_MESSAGE(out.size() == 32, "Error wrong result size when h256::ref().copyTo(dev::bytesRef out)");
	BOOST_CHECK_MESSAGE(out.toBytes() == originalSequence, "Error when h256::ref().copyTo(dev::bytesRef out)");
}

BOOST_AUTO_TEST_SUITE_END()
