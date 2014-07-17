env.note('Creating Config...')
var configCode = eth.lll("
{
  [[69]] (caller)
  (returnlll {
    (when (&& (= (calldatasize) 64) (= (caller) @@69))
      (for {} (< @i (calldatasize)) [i](+ @i 64)
        [[ (calldataload @i) ]] (calldataload (+ @i 32))
      )
    )
    (return @@ $0)
  })
}
")
env.note('Config code: ' + configCode.unbin())
var config = "0x9ef0f0d81e040012600b0c1abdef7c48f720f88a";
eth.create(eth.key, '0', configCode, 10000, eth.gasPrice, function(a) { config = a; })

env.note('Config at address ' + config)

var nameRegCode = eth.lll("
{
  [[(address)]] 'NameReg
  [['NameReg]] (address)
  [[" + config + "]] 'Config
  [['Config]] " + config + "
  [[69]] (caller)
  (returnlll {
    (when (= $0 'register) {
      (when @@ $32 (stop))
      (when @@(caller) [[@@(caller)]] 0)
      [[$32]] (caller)
      [[(caller)]] $32
      (stop)
    })
    (when (&& (= $0 'unregister) @@(caller)) {
      [[@@(caller)]] 0
      [[(caller)]] 0
      (stop)
    })
    (when (&& (= $0 'kill) (= (caller) @@69)) (suicide (caller)))
    (return @@ $0)
  })
}
");
env.note('NameReg code: ' + nameRegCode.unbin())

var nameReg = "0x3face8f2b3ef580265f0f67a57ce0fb78b135613";
env.note('Create NameReg...')
eth.create(eth.key, '0', nameRegCode, 10000, eth.gasPrice, function(a) { nameReg = a; })

env.note('NameReg at address ' + nameReg)

env.note('Register NameReg...')
eth.transact(eth.key, '0', config, "0".pad(32) + nameReg.pad(32), 10000, eth.gasPrice);

var coinsCode = eth.lll("
{
[0]'register [32]'Coins
(msg allgas " + nameReg + " 0 0 64)
(returnlll {
	(def 'name $0)
	(def 'address (caller))
	(when (|| (& 0xffffffffffffffffffffffffff name) @@name) (stop))
	(set 'n (+ @@0 1))
	[[0]] @n
	[[@n]] name
	[[name]] address
})
}
");

var coins;
env.note('Create Coins...')
eth.create(eth.key, '0', coinsCode, 10000, eth.gasPrice, function(a) { coins = a; })

env.note('Coins at address ' + coins)

env.note('Register Coins...')
eth.transact(eth.key, '0', config, "1".pad(32) + coins.pad(32), 10000, eth.gasPrice);

var gavCoinCode = eth.lll("
{
[[ (caller) ]] 0x1000000
[[ 0x69 ]] (caller)
[[ 0x42 ]] (number)

[0]'register [32]'GavCoin
(msg allgas " + nameReg + " 0 0 64)
(msg " + coins + " 'GAV)

(returnlll {
	(when (&& (= $0 'kill) (= (caller) @@0x69)) (suicide (caller)))
	(when (= $0 'balance) (return @@$32))
	(when (= $0 'approved) (return @@ @(sha3 (^ (if (= (calldatasize) 64) (caller) $64) $32))) )
	
	(when (= $0 'approve) {
		[[@(sha3 (^ (caller) $32))]] $32
		(stop)
	})

	(when (= $0 'send) {
		(set 'fromVar (if (= (calldatasize) 96)
			(caller)
			{
				(when (! @@ @(sha3 (^ $96 $32)) ) (stop))
				$96
			}
		))
		(def 'to $32)
		(def 'value $64)
		(def 'from (get 'fromVar))
		(set 'fromBal @@from)
		(when (< @fromBal value) (stop))
		[[ from ]]: (- @fromBal value)
		[[ to ]]: (+ @@to value)
		(stop)
	})

	(set 'n @@0x42)
	(when (&& (|| (= $0 'mine) (! (calldatasize))) (> (number) @n)) {
		[[(coinbase)]] (+ @@(coinbase) 1024)
		[[0x42]] (+ @n 1)
	})

	(return @@ $0)
})
}
");

var gavCoin;
env.note('Create GavCoin...')
eth.create(eth.key, '0', gavCoinCode, 10000, eth.gasPrice, function(a) { gavCoin = a; });

env.note('Register GavCoin...')
eth.transact(eth.key, '0', config, "2".pad(32) + gavCoin.pad(32), 10000, eth.gasPrice);

env.note('Register my name...')
eth.transact(eth.key, '0', nameReg, "register".pad(32) + "Gav".pad(32), 10000, eth.gasPrice);

env.note('All done.')

// env.load('/home/gav/Eth/cpp-ethereum/stdserv.js')
