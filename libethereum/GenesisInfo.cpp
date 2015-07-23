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
/** @file GenesisInfo.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "GenesisInfo.h"

std::string const dev::eth::c_genesisInfo =
R"ETHEREUM(
{
	"nonce": "0x000000000000002a",
	"difficulty": "0x20000",
	"mixhash": "0x0000000000000000000000000000000000000000000000000000000000000000",
	"coinbase": "0x0000000000000000000000000000000000000000",
	"timestamp": "0x00",
	"parentHash": "0x0000000000000000000000000000000000000000000000000000000000000000",
	"extraData": "0x",
	"gasLimit": "0x2FEFD8",
	"alloc": {
		"0000000000000000000000000000000000000001": { "wei": "1" },
		"0000000000000000000000000000000000000002": { "wei": "1" },
		"0000000000000000000000000000000000000003": { "wei": "1" },
		"0000000000000000000000000000000000000004": { "wei": "1" },
		"dbdbdb2cbd23b783741e8d7fcf51e459b497e4a6": { "wei": "1606938044258990275541962092341162602522202993782792835301376" },
		"e6716f9544a56c530d868e4bfbacb172315bdead": { "wei": "1606938044258990275541962092341162602522202993782792835301376" },
		"b9c015918bdaba24b4ff057a92a3873d6eb201be": { "wei": "1606938044258990275541962092341162602522202993782792835301376" },
		"1a26338f0d905e295fccb71fa9ea849ffa12aaf4": { "wei": "1606938044258990275541962092341162602522202993782792835301376" },
		"2ef47100e0787b915105fd5e3f4ff6752079d5cb": { "wei": "1606938044258990275541962092341162602522202993782792835301376" },
		"cd2a3d9f938e13cd947ec05abc7fe734df8dd826": { "wei": "1606938044258990275541962092341162602522202993782792835301376" },
		"6c386a4b26f73c802f34673f7248bb118f97424a": { "wei": "1606938044258990275541962092341162602522202993782792835301376" },
		"e4157b34ea9615cfbde6b4fda419828124b70c78": { "wei": "1606938044258990275541962092341162602522202993782792835301376" }
	}
}
)ETHEREUM";
