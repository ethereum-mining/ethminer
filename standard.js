///TODO

var compile = function(name) { return web3.eth.solidity(env.contents("/home/gav/Eth/dapp-bin/" + name + "/" + name + ".sol")); web3.eth.flush(); };
var create = function(code) { return web3.eth.transact({ 'code': code }); web3.eth.flush(); };
var createVal = function(code, val) { return web3.eth.transact(val ? { 'code': code, 'value': val } : { 'code': code }); web3.eth.flush(); };
var send = function(from, val, to) { web3.eth.transact({ 'from': from, 'value': val, 'to': to }); web3.eth.flush(); };
var initService = function(name) { return create(compile(name)); };
var initServiceVal = function(name, val) { createVal(compile(name), val); };

var addrConfig = create(compile("config"));
var addrNameReg = initService("namereg");
var addrGavsino = initServiceVal("gavmble", "1000000000000000000");
var addrCoinReg = initService("coins");

var abiNameReg = [{"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"getName","outputs":[{"name":"o_name","type":"string32"}]},{"constant":false,"inputs":[{"name":"name","type":"string32"}],"name":"register","outputs":[]},{"constant":true,"inputs":[{"name":"name","type":"string32"}],"name":"addressOf","outputs":[{"name":"addr","type":"address"}]},{"constant":true,"inputs":[{"name":"_name","type":"string32"}],"name":"getAddress","outputs":[{"name":"o_owner","type":"address"}]},{"constant":false,"inputs":[],"name":"unregister","outputs":[]},{"constant":true,"inputs":[{"name":"addr","type":"address"}],"name":"nameOf","outputs":[{"name":"name","type":"string32"}]}];
var regName = function(account, name) { return web3.eth.contract(addrNameReg, abiNameReg).transact({'from': account, 'gas': 10000}).register(name); };

send(web3.eth.accounts[0], '100000000000000000000', web3.eth.accounts[1]);
regName(web3.eth.accounts[0], 'Gav');
regName(web3.eth.accounts[1], 'Gav Would');

/*
// ASYNC API

var compile = function(name) { return web3.eth.solidity(env.contents("/home/gav/Eth/dapp-bin/" + name + "/" + name + ".sol")); };
var create = function(code) { return web3.eth.transact({ 'code': code }); };
var createVal = function(code, val) { return web3.eth.transact(val ? { 'code': code, 'value': val } : { 'code': code }); };
var send = function(from, val, to) { return web3.eth.transact({ 'from': from, 'value': val, 'to': to }); };
var initService = function(name, dep) { return dep.then(function(){ return compile(name).then(create); }); };
var initServiceVal = function(name, dep, val) { return dep.then(function(){ return compile(name).then(function(c) { createVal(c, val); }); }); };

var addrConfig = compile("config").then(create);
var addrNameReg = initService("namereg", addrConfig);
var addrGavsino = initServiceVal("gavmble", addrNameReg, "1000000000000000000");

var abiNameReg = [{"constant":true,"inputs":[{"name":"name","type":"string32"}],"name":"addressOf","outputs":[{"name":"addr","type":"address"}]},{"constant":false,"inputs":[],"name":"kill","outputs":[]},{"constant":true,"inputs":[{"name":"addr","type":"address"}],"name":"nameOf","outputs":[{"name":"name","type":"string32"}]},{"constant":false,"inputs":[{"name":"name","type":"string32"}],"name":"register","outputs":[]},{"constant":false,"inputs":[],"name":"unregister","outputs":[]}];
var regName = function(account, name) { return web3.contract(addrNameReg, abiNameReg).register(name).transact({'from': account, 'gas': 10000}); };

//var coins = initService("coins", 1, nameReg);
//var coin = initService("coin", 2, coins);
//var approve = function(account, approvedAddress) { web3.eth.transact({ 'from': account, 'to': coin, 'gas': '10000', 'data': [ web3.fromAscii('approve'), approvedAddress ] }); };
//var exchange = initService("exchange", 3, coin);
//var offer = function(account, haveCoin, haveVal, wantCoin, wantVal) { web3.eth.transact({ 'from': account, 'to': exchange, 'gas': '10000', 'data': [web3.fromAscii('new'), haveCoin, haveVal, wantCoin, wantVal] }); };

addrConfig.then(function() {
    env.note("config ready");
    web3.eth.accounts.then(function(accounts)
	{
        env.note("accounts ready");
		var funded = send(accounts[0], '100000000000000000000', accounts[1]);
        funded.then(function(){
            env.note("second account funded");
            regName(accounts[1], 'Gav Would');
        });
		regName(accounts[0], 'Gav');
		
        // TODO: once we have the exchange.
//		approve(accounts[0], exchange).then(function(){ offer(accounts[0], coin, '5000', '0', '5000000000000000000'); });
//      funded.then(function(){ approve(accounts[1], exchange); });

		// TODO: once we have a new implementation of DNSReg.
		//	env.note('Register gav.eth...')
		//	eth.transact({ 'to': dnsReg, 'data': [web3.fromAscii('register'), web3.fromAscii('gav'), web3.fromAscii('opensecrecy.com')] });
	});
});
*/
// TODO
/*
var nameRegJeff;

env.note('Create NameRegJeff...')
eth.transact({ 'code': nameRegCode }, function(a) { nameRegJeff = a; });

env.note('Register NameRegJeff...')
eth.transact({ 'to': config, 'data': ['4', nameRegJeff] });

var dnsRegCode = '0x60006000546000600053602001546000600053604001546020604060206020600073661005d2720d855f1d9976f88bb10c1a3398c77f6103e8f17f7265676973746572000000000000000000000000000000000000000000000000600053606001600060200201547f446e735265670000000000000000000000000000000000000000000000000000600053606001600160200201546000600060006000604060606000600053604001536103e8f1327f6f776e65720000000000000000000000000000000000000000000000000000005761011663000000e46000396101166000f20060006000547f72656769737465720000000000000000000000000000000000000000000000006000602002350e0f630000006d596000600160200235560e0f630000006c59600032560e0f0f6300000057596000325657600260200235600160200235576001602002353257007f64657265676973746572000000000000000000000000000000000000000000006000602002350e0f63000000b95960016020023532560e0f63000000b959600032576000600160200235577f6b696c6c000000000000000000000000000000000000000000000000000000006000602002350e0f630000011559327f6f776e6572000000000000000000000000000000000000000000000000000000560e0f63000001155932ff00';

var dnsReg;
env.note('Create DnsReg...')
eth.transact({ 'code': dnsRegCode }, function(a) { dnsReg = a; });

env.note('DnsReg at address ' + dnsReg)

env.note('Register DnsReg...')
eth.transact({ 'to': config, 'data': ['4', dnsReg] });
*/

// env.load('/home/gav/Eth/cpp-ethereum/stdserv.js')
