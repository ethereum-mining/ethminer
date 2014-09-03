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
/** @file Common.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <string>
#include <chrono>
#include <libethential/Common.h>
#include <libethential/Log.h>
#include <libethcore/CommonEth.h>
#include <libp2p/Common.h>

namespace shh
{

using h256 = eth::h256;
using h512 = eth::h512;
using h256s = eth::h256s;
using bytes = eth::bytes;
using RLPStream = eth::RLPStream;
using RLP = eth::RLP;
using bytesRef = eth::bytesRef;
using bytesConstRef = eth::bytesConstRef;
using h256Set = eth::h256Set;

class WhisperHost;
class WhisperPeer;
class Whisper;

enum WhisperPacket
{
	StatusPacket = 0x20,
	MessagesPacket,
	AddFilterPacket,
	RemoveFilterPacket
};

}
