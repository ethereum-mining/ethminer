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
var addrGavCoin = initService("coin");
var addrRegistrar = initService("registrar");

var abiNameReg = [{"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"getName","outputs":[{"name":"o_name","type":"string32"}]},{"constant":false,"inputs":[{"name":"name","type":"string32"}],"name":"register","outputs":[]},{"constant":true,"inputs":[{"name":"name","type":"string32"}],"name":"addressOf","outputs":[{"name":"addr","type":"address"}]},{"constant":true,"inputs":[{"name":"_name","type":"string32"}],"name":"getAddress","outputs":[{"name":"o_owner","type":"address"}]},{"constant":false,"inputs":[],"name":"unregister","outputs":[]},{"constant":true,"inputs":[{"name":"addr","type":"address"}],"name":"nameOf","outputs":[{"name":"name","type":"string32"}]}];
var regName = function(account, name) { return web3.eth.contract(addrNameReg, abiNameReg).transact({'from': account, 'gas': 10000}).register(name); };

send(web3.eth.accounts[0], '100000000000000000000', web3.eth.accounts[1]);
regName(web3.eth.accounts[0], 'Gav');
regName(web3.eth.accounts[1], 'Gav Would');

// TODO: once we have the exchange.
//  var offer = function(account, haveCoin, haveVal, wantCoin, wantVal) { web3.eth.transact({ 'from': account, 'to': exchange, 'gas': '10000', 'data': [web3.fromAscii('new'), haveCoin, haveVal, wantCoin, wantVal] }); };
//  approve(accounts[0], exchange).then(function(){ offer(accounts[0], coin, '5000', '0', '5000000000000000000'); });
//  funded.then(function(){ approve(accounts[1], exchange); });

// TODO: once we have a new implementation of DNSReg.
//  env.note('Register gav.eth...')
//  eth.transact({ 'to': dnsReg, 'data': [web3.fromAscii('register'), web3.fromAscii('gav'), web3.fromAscii('opensecrecy.com')] });
