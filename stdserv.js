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

var dnsRegCode = '0x60006000546000600053602001546000600053604001546020604060206020600073661005d2720d855f1d9976f88bb10c1a3398c77f6103e8f17f7265676973746572000000000000000000000000000000000000000000000000600053606001600060200201547f446e735265670000000000000000000000000000000000000000000000000000600053606001600160200201546000600060006000604060606000600053604001536103e8f1327f6f776e65720000000000000000000000000000000000000000000000000000005761011663000000e46000396101166000f20060006000547f72656769737465720000000000000000000000000000000000000000000000006000602002350e0f630000006d596000600160200235560e0f630000006c59600032560e0f0f6300000057596000325657600260200235600160200235576001602002353257007f64657265676973746572000000000000000000000000000000000000000000006000602002350e0f63000000b95960016020023532560e0f63000000b959600032576000600160200235577f6b696c6c000000000000000000000000000000000000000000000000000000006000602002350e0f630000011559327f6f776e6572000000000000000000000000000000000000000000000000000000560e0f63000001155932ff00';

var dnsReg;
env.note('Create DnsReg...')
eth.create(eth.key, '0', dnsRegCode, 10000, eth.gasPrice, function(a) { dnsReg = a; })

env.note('DnsReg at address ' + dnsReg)

env.note('Register DnsReg...')
eth.transact(eth.key, '0', config, "4".pad(32) + dnsReg.pad(32), 10000, eth.gasPrice);

var coinRegCode = eth.lll("
{
[0]'register [32]'CoinReg
(msg allgas " + nameReg + " 0 0 64)
(returnlll {
	(def 'name $0)
	(def 'denom $32)
	(def 'address (caller))
	(when (|| (& 0xffffffffffffffffffffffffff name) @@name) (stop))
	(set 'n (+ @@0 1))
	[[0]] @n
	[[@n]] name
	[[name]] address
	[[(sha3 name)]] denom
})
}
");

var coinReg;
env.note('Create CoinReg...')
eth.create(eth.key, '0', coinRegCode, 10000, eth.gasPrice, function(a) { coinReg = a; })

env.note('CoinReg at address ' + coinReg)

env.note('Register CoinReg...')
eth.transact(eth.key, '0', config, "1".pad(32) + coinReg.pad(32), 10000, eth.gasPrice);

var gavCoinCode = eth.lll("
{
[[ (caller) ]] 0x1000000
[[ 0x69 ]] (caller)
[[ 0x42 ]] (number)

[0]'register [32]'GavCoin
(msg allgas " + nameReg + " 0 0 64)
[0]'GAV [32] 1000
(msg allgas " + coinReg + " 0 0 64)

(returnlll {
	(when (&& (= $0 'kill) (= (caller) @@0x69)) (suicide (caller)))
	(when (= $0 'balance) (return @@$32))
	(when (= $0 'approved) (return @@ (sha3pair (if (= (calldatasize) 64) (caller) $64) $32)) )
	
	(when (= $0 'approve) {
		[[(sha3pair (caller) $32)]] $32
		(stop)
	})

	(when (= $0 'send) {
		(set 'fromVar (if (= (calldatasize) 96)
			(caller)
			{
				(when (! @@ (sha3pair (origin) (caller))) (return 0))
				(origin)
			}
		))
		(def 'to $32)
		(def 'value $64)
		(def 'from (get 'fromVar))
		(set 'fromBal @@from)
		(when (< @fromBal value) (return 0))
		[[ from ]]: (- @fromBal value)
		[[ to ]]: (+ @@to value)
		(return 1)
	})

	(set 'n @@0x42)
	(when (&& (|| (= $0 'mine) (! (calldatasize))) (> (number) @n)) {
		(set 'b (- (number) @n))
		[[(coinbase)]] (+ @@(coinbase) (* 1000 @b))
		[[(caller)]] (+ @@(caller) (* 1000 @b))
		[[0x42]] (number)
		(return @b)
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

var exchangeCode = eth.lll("
{
[0] 'register
[32] 'Exchange
(msg allgas 0x50441127ea5b9dfd835a9aba4e1dc9c1257b58ca 0 0 64)

(def 'min (a b) (if (< a b) a b))

(def 'head (_list) @@ _list)
(def 'next (_item) @@ _item)
(def 'inc (itemref) [itemref]: (next @itemref))
(def 'rateof (_item) @@ (+ _item 1))
(def 'idof (_item) @@ (+ _item 2))
(def 'wantof (_item) @@ (+ _item 3))
(def 'newitem (rate who want list) {
	(set 'pos (sha3trip rate who list))
	[[ (+ @pos 1) ]] rate
	[[ (+ @pos 2) ]] who
	[[ (+ @pos 3) ]] want
	@pos
})
(def 'stitchitem (parent pos) {
	[[ pos ]] @@ parent
	[[ parent ]] pos
})
(def 'addwant (_item amount) [[ (+ _item 3) ]] (+ @@ (+ _item 3) amount))
(def 'deductwant (_item amount) [[ (+ _item 3) ]] (- @@ (+ _item 3) amount))

(def 'xfer (contract to amount)
	(if contract {
		[0] 'send
		[32] to
		[64] amount
		(msg allgas contract 0 0 96)
	}
		(send to amount)
	)
)

(def 'fpdiv (a b) (/ (+ (/ b 2) (* a (exp 2 128))) b))
(def 'fpmul (a b) (/ (* a b) (exp 2 128)) )

(returnlll {
	(when (= $0 'new) {
		(set 'offer $32)
		(set 'xoffer (if @offer $64 (callvalue)))
		(set 'want $96)
		(set 'xwant $128)
		(set 'rate (fpdiv @xoffer @xwant))
		(set 'irate (fpdiv @xwant @xoffer))

		(unless (&& @rate @irate @xoffer @xwant) (stop))

		(when @offer {
			(set 'arg1 'send)
			(set 'arg2 (address))
			(set 'arg3 @xoffer)
			(set 'arg4 (caller))
			(unless (msg allgas @offer 0 arg1 128) (stop))
		})
		(set 'list (sha3pair @offer @want))
		(set 'ilist (sha3pair @want @offer))

		(set 'last @ilist)
		(set 'item @@ @last)
		
		(for {} (&& @item (>= (rateof @item) @irate)) {} {
			(set 'offerA (min @xoffer (wantof @item)))
			(set 'wantA (fpmul @offerA (rateof @item)))

			(set 'xoffer (- @xoffer @offerA))
			(set 'xwant (- @xwant @wantA))

			(deductwant @item @offerA)

			(xfer @offer (idof @item) @offerA)
			(xfer @want (caller) @wantA)

			(unless @xoffer (stop))

			(set 'item @@ @item)
			[[ @last ]] @item
		})

		(set 'last @list)
		(set 'item @@ @last)
		
		(set 'newpos (newitem @rate (caller) @xwant @list))

		(for {} (&& @item (!= @item @newpos) (>= (rateof @item) @rate)) { (set 'last @item) (inc item) } {})
		(if (= @item @newpos)
			(addwant @item @wantx)
			(stitchitem @last @newpos)
		)
		(stop)
	})
	(when (= $0 'delete) {
		(set 'offer $32)
		(set 'want $64)
		(set 'rate $96)
		(set 'list (sha3pair @offer @want))
		(set 'last @list)
		(set 'item @@ @last)
		(for {} (&& @item (!= (idof @item) (caller)) (!= (rateof @item) @rate)) { (set 'last @item) (inc item) } {})
		(when @item {
			(set 'xoffer (fpmul (wantof @item) (rateof @item)))
			[[ @last ]] @@ @item
			(xfer @offer (caller) @xoffer)
		})
		(stop)
	})
	(when (= $0 'price) {
		(set 'offer $32)
		(set 'want $96)
		(set 'item (head (sha3pair @offer @want)))
		(return (if @item (rateof @list) 0))
	})
})
}
");

var exchange;
env.note('Create Exchange...')
eth.create(eth.key, '0', exchangeCode, 10000, eth.gasPrice, function(a) { exchange = a; });

env.note('Register Exchange...')
eth.transact(eth.key, '0', config, "3".pad(32) + exchange.pad(32), 10000, eth.gasPrice);




env.note('Register my name...')
eth.transact(eth.key, '0', nameReg, "register".pad(32) + "Gav".pad(32), 10000, eth.gasPrice);

env.note('Dole out ETH to other address...')
eth.transact(eth.key, '100000000000000000000', eth.secretToAddress(eth.keys[1]), "", 10000, eth.gasPrice);

env.note('Register my other name...')
eth.transact(eth.keys[1], '0', nameReg, "register".pad(32) + "Gav Would".pad(32), 10000, eth.gasPrice);

env.note('Approve Exchange...')
eth.transact(eth.key, '0', gavCoin, "approve".pad(32) + exchange.pad(32), 10000, eth.gasPrice);

env.note('Approve Exchange on other address...')
eth.transact(eth.keys[1], '0', gavCoin, "approve".pad(32) + exchange.pad(32), 10000, eth.gasPrice);

env.note('Make offer 5000GAV/5ETH...')
eth.transact(eth.key, '0', exchange, "new".pad(32) + gavCoin.pad(32) + "5000".pad(32) + "0".pad(32) + "5000000000000000000".pad(32), 10000, eth.gasPrice);

env.note('All done.')

// env.load('/home/gav/Eth/cpp-ethereum/stdserv.js')
