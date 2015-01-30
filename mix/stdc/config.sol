//sol Config
// Simple global configuration registrar.
// @authors:
//   Gav Wood <g@ethdev.com>
#require mortal
contract Config is mortal {
	function register(uint id, address service) {
		if (tx.origin != owner)
			return;
		services[id] = service;
		log1(0, id);
	}

	function unregister(uint id) {
		if (msg.sender != owner && services[id] != msg.sender)
			return;
		services[id] = address(0);
		log1(0, id);
	}

	function lookup(uint service) constant returns(address a) {
		return services[service];
	}

	mapping (uint => address) services;
}

/*

// Solidity Interface:
contract Config{function lookup(uint256 service)constant returns(address a){}function kill(){}function unregister(uint256 id){}function register(uint256 id,address service){}}

// Example Solidity use:
address addrConfig = 0xf025d81196b72fba60a1d4dddad12eeb8360d828;
address addrNameReg = Config(addrConfig).lookup(1);

// JS Interface:
var abiConfig = [{"constant":false,"inputs":[],"name":"kill","outputs":[]},{"constant":true,"inputs":[{"name":"service","type":"uint256"}],"name":"lookup","outputs":[{"name":"a","type":"address"}]},{"constant":false,"inputs":[{"name":"id","type":"uint256"},{"name":"service","type":"address"}],"name":"register","outputs":[]},{"constant":false,"inputs":[{"name":"id","type":"uint256"}],"name":"unregister","outputs":[]}];

// Example JS use:
var addrConfig = "0x661005d2720d855f1d9976f88bb10c1a3398c77f";
var addrNameReg;
web3.eth.contract(addrConfig, abiConfig).lookup(1).call().then(function(r){ addrNameReg = r; })

*/
