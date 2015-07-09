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
/** @file ProofOfWork.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "ProofOfWork.h"
#include "BlockInfo.h"
using namespace std;
using namespace dev;
using namespace eth;

const Address BasicAuthority::Authority = Address("1234567890123456789012345678901234567890");

bool BasicAuthority::verify(BlockInfo const& _header)
{
	return toAddress(recover(_header.proof.sig, _header.headerHash(WithoutProof))) == Authority;
}

bool BasicAuthority::preVerify(BlockInfo const& _header)
{
	return SignatureStruct(_header.proof.sig).isValid();
}

BasicAuthority::WorkPackage BasicAuthority::package(BlockInfo const& _header)
{
	return WorkPackage{_header.headerHash(WithoutProof)};
}

void BasicAuthority::Farm::sealBlock(BlockInfo const& _bi)
{
	m_onSolutionFound(Solution{sign(m_secret, _bi.headerHash(WithoutProof))});
}

