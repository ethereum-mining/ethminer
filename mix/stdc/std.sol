//TODO: use imports
contract owned{function owned(){owner = msg.sender;}modifier onlyowner(){if(msg.sender==owner)_}address owner;}
contract mortal is owned {function kill() { if (msg.sender == owner) suicide(owner); }}

//sol Config
// Simple global configuration registrar.
// @authors:
//   Gav Wood <g@ethdev.com>


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

//sol NameReg
// Simple global name registrar.
// @authors:
//   kobigurk (from #ethereum-dev)
//   Gav Wood <g@ethdev.com>

contract NameRegister {
	function getAddress(string32 _name) constant returns (address o_owner) {}
	function getName(address _owner) constant returns (string32 o_name) {}
}

contract NameReg is owned, NameRegister {
	function NameReg() {
		address addrConfig = 0xf025d81196b72fba60a1d4dddad12eeb8360d828;
		toName[addrConfig] = "Config";
		toAddress["Config"] = addrConfig;
		toName[this] = "NameReg";
		toAddress["NameReg"] = this;
		Config(addrConfig).register(1, this);
		log1(0, hash256(addrConfig));
		log1(0, hash256(this));
	}

	function register(string32 name) {
		// Don't allow the same name to be overwritten.
		if (toAddress[name] != address(0))
			return;
		// Unregister previous name if there was one.
		if (toName[msg.sender] != "")
			toAddress[toName[msg.sender]] = 0;

		toName[msg.sender] = name;
		toAddress[name] = msg.sender;
		log1(0, hash256(msg.sender));
	}

	function unregister() {
		string32 n = toName[msg.sender];
		if (n == "")
			return;
		log1(0, hash256(toAddress[n]));
		toName[msg.sender] = "";
		toAddress[n] = address(0);
	}

	function addressOf(string32 name) constant returns (address addr) {
		return toAddress[name];
	}

	function nameOf(address addr) constant returns (string32 name) {
		return toName[addr];
	}

	mapping (address => string32) toName;
	mapping (string32 => address) toAddress;
}


/*

// Solidity Interface:
contract NameReg{function kill(){}function register(string32 name){}function addressOf(string32 name)constant returns(address addr){}function unregister(){}function nameOf(address addr)constant returns(string32 name){}}

// Example Solidity use:
NameReg(addrNameReg).register("Some Contract");

// JS Interface:
var abiNameReg = [{"constant":true,"inputs":[{"name":"name","type":"string32"}],"name":"addressOf","outputs":[{"name":"addr","type":"address"}]},{"constant":false,"inputs":[],"name":"kill","outputs":[]},{"constant":true,"inputs":[{"name":"addr","type":"address"}],"name":"nameOf","outputs":[{"name":"name","type":"string32"}]},{"constant":false,"inputs":[{"name":"name","type":"string32"}],"name":"register","outputs":[]},{"constant":false,"inputs":[],"name":"unregister","outputs":[]}];

// Example JS use:
web3.eth.contract(addrNameReg, abiNameReg).register("My Name").transact();

*/
