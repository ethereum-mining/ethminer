# Pool Examples for ETH

Pool connection definition is issued via `-P` argument which has this syntax:

```
-P scheme://user[.workername][:password]@hostname:port[/...]
```
__values in square brackets are optional__

where `scheme` can be any of:

* `http` for getwork mode (geth)
* `stratum+tcp` for plain stratum mode
* `stratum1+tcp` for plain stratum eth-proxy compatible mode
* `stratum2+tcp` for plain stratum NiceHash compatible mode

## Secure socket comunications for stratum only

Ethminer supports secure socket communications (where pool implements and offers it) to avoid the risk of a [man-in-the-middle attack](https://en.wikipedia.org/wiki/Man-in-the-middle_attack)
To enable it simply replace tcp with either:

* `tls` to enable secure socket communication
* `ssl` or `tls12` to enable secure socket communication **allowing only TLS 1.2** encryption

thus your connection scheme changes to `-P stratum+tls://[...]` or `-P stratum+tls12://[...]`. Same applies for `stratum1` and `stratum2`.

## Only for version 0.16+ (older versions not affected)

Stratum autodetection has been introduced to mitigate user's duty to guess/find which stratum flavour to apply (stratum or stratum1 or stratum2).
If you want to let ethminer do the tests for you simply enter scheme as `stratum://` (note `+tcp` is missing) or `stratums://` for secure socket or `stratumss://` for secure socket **allowing only TLS 1.2** encryption.

## Common samples

Here you can find a collection of samples to connect to most commonly used ethash pools. (alphabetic order).

* Stratum connection is **always to be preferred** over **getwork** when pool offers it due to its better network latency.
* If possible the samples use a protocol which supports reporting of hashrate (`--report-hashrate`) if pool supports this.

**Check for updates in the pool connection settings visiting the pools homepage.**

## Variables

We tried to merge the requirements of the variables so they match all pools.

|   Variables  | Description  | Sample |
| ------------ | ------------ | ------ |
| `ETH_WALLET` | Replace `ETH_WALLET` with your Ethereum wallet number including the leading `0x`.                                                                          | `0x1234567890ABCDEF1234567890abcdef12345678` |
| `WORKERNAME` | `WORKERNAME` may only contain letters and numbers. Some pools also only allow up to a maximum of 8 characters!                                             | `pl1rig01` |
| `EMAIL`      | `EMAIL` may contain letters, numbers, underscores, dashes, dots and the @-sign. It **must** contain a @-sign and a dot!                                    | `joe1.doe_jr-ny@acme.com` |
| `USERNAME`   | `USERNAME` you got from the pool (like [miningpoolhub.com](#miningpoolhubcom))                                                                             | `my_username` |
| `WORKERPWD`  | `WORKERPWD` is the password you got from the pool for the worker (like [miningpoolhub.com](#miningpoolhubcom)) - if you have no password set try using 'x' | `my_workerpwd` |
| `BTC_WALLET` | As some pools honor your work in BTC (eg [nicehash.com](#nicehashcom)) `BTC_WALLET` is your Bitcoin wallet address                                         | `1A2b3C4d5e5F6g7H8I9j0kLmNoPqRstUvW` |

## Servers

The servers are listed in alphabetical order. To get best results reorder them from nearest to farest distance depending on your geographic location.

## Pools (alphabetic order)

| Pool Name | Pool Homepage | Details about connection |
| --------- | ------------- | - |
| [2miners.com](#2minerscom) | <https://2miners.com/> | <https://eth.2miners.com/en/help> |
| [dwarfpool.org](#dwarfpoolorg) | <https://dwarfpool.com/> | <https://dwarfpool.com/eth> |
| [ethermine.org](#ethermineorg) | <https://ethermine.org/> | <https://ethermine.org/> |
| [ethpool.org](#ethpoolorg) | <https://www.ethpool.org/> | <https://www.ethpool.org/> |
| [f2pool.com](#f2poolcom) | <https://www.f2pool.com/> | <https://www.f2pool.com/help/?#tab-content-eth> |
| [miningpoolhub.com](#miningpoolhubcom) | <https://miningpoolhub.com/> | <https://ethereum.miningpoolhub.com/> |
| [nanopool.org](#nanopoolorg) | <https://nanopool.org/> | <https://eth.nanopool.org/help> |
| [nicehash.com](#nicehashcom) | <https://www.nicehash.com/> | <https://www.nicehash.com/help/which-stratum-servers-are-available> |
| [sparkpool.com](#sparkpoolcom) | <https://sparkpool.com/> | <https://eth.sparkpool.com/> |
| [whalesburg.com](#whalesburgcom) | <https://whalesburg.com/> | <https://whalesburg.com/start_mining/> |

### 2miners.com

```
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth.2miners.com:2020
```

### dwarfpool.org

With email

```
-P stratum1+tcp://ETH_WALLET@eth-ar.dwarfpool.com:8008/WORKERNAME/EMAIL
-P stratum1+tcp://ETH_WALLET@eth-asia.dwarfpool.com:8008/WORKERNAME/EMAIL
-P stratum1+tcp://ETH_WALLET@eth-au.dwarfpool.com:8008/WORKERNAME/EMAIL
-P stratum1+tcp://ETH_WALLET@eth-br.dwarfpool.com:8008/WORKERNAME/EMAIL
-P stratum1+tcp://ETH_WALLET@eth-cn.dwarfpool.com:8008/WORKERNAME/EMAIL
-P stratum1+tcp://ETH_WALLET@eth-cn2.dwarfpool.com:8008/WORKERNAME/EMAIL
-P stratum1+tcp://ETH_WALLET@eth-eu.dwarfpool.com:8008/WORKERNAME/EMAIL
-P stratum1+tcp://ETH_WALLET@eth-hk.dwarfpool.com:8008/WORKERNAME/EMAIL
-P stratum1+tcp://ETH_WALLET@eth-sg.dwarfpool.com:8008/WORKERNAME/EMAIL
-P stratum1+tcp://ETH_WALLET@eth-ru.dwarfpool.com:8008/WORKERNAME/EMAIL
-P stratum1+tcp://ETH_WALLET@eth-ru2.dwarfpool.com:8008/WORKERNAME/EMAIL
-P stratum1+tcp://ETH_WALLET@eth-us.dwarfpool.com:8008/WORKERNAME/EMAIL
-P stratum1+tcp://ETH_WALLET@eth-us2.dwarfpool.com:8008/WORKERNAME/EMAIL
```

Without email

```
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth-ar.dwarfpool.com:8008
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth-asia.dwarfpool.com:8008
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth-au.dwarfpool.com:8008
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth-br.dwarfpool.com:8008
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth-cn.dwarfpool.com:8008
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth-cn2.dwarfpool.com:8008
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth-eu.dwarfpool.com:8008
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth-hk.dwarfpool.com:8008
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth-sg.dwarfpool.com:8008
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth-ru.dwarfpool.com:8008
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth-ru2.dwarfpool.com:8008
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth-us.dwarfpool.com:8008
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth-us2.dwarfpool.com:8008
```

### ethermine.org

Non-SSL connection:

```
-P stratum1+tcp://ETH_WALLET.WORKERNAME@asia1.ethermine.org:4444
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eu1.ethermine.org:4444
-P stratum1+tcp://ETH_WALLET.WORKERNAME@us1.ethermine.org:4444
-P stratum1+tcp://ETH_WALLET.WORKERNAME@us2.ethermine.org:4444
```

SSL connection:

```
-P stratum1+ssl://ETH_WALLET.WORKERNAME@asia1.ethermine.org:5555
-P stratum1+ssl://ETH_WALLET.WORKERNAME@eu1.ethermine.org:5555
-P stratum1+ssl://ETH_WALLET.WORKERNAME@us1.ethermine.org:5555
-P stratum1+ssl://ETH_WALLET.WORKERNAME@us2.ethermine.org:5555
```

### ethpool.org

 ```
 -P stratum1+tcp://ETH_WALLET.WORKERNAME@asia1.ethpool.org:3333
 -P stratum1+tcp://ETH_WALLET.WORKERNAME@eu1.ethpool.org:3333
 -P stratum1+tcp://ETH_WALLET.WORKERNAME@us1.ethpool.org:3333
 ```

### f2pool.com

```
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth.f2pool.com:8008
```

### miningpoolhub.com

```
-P stratum2+tcp://USERNAME.WORKERNAME:WORKERPWD@asia.ethash-hub.miningpoolhub.com:20535
-P stratum2+tcp://USERNAME.WORKERNAME:WORKERPWD@europe.ethash-hub.miningpoolhub.com:20535
-P stratum2+tcp://USERNAME.WORKERNAME:WORKERPWD@us-east.ethash-hub.miningpoolhub.com:20535
```

HINT: It seems the password is not being verified by the pool so you can use a plain `x` as `WORKERPWD`.

### nanopool.org

With email:

```
-P stratum1+tcp://ETH_WALLET@eth-asia1.nanopool.org:9999/WORKERNAME/EMAIL
-P stratum1+tcp://ETH_WALLET@eth-eu1.nanopool.org:9999/WORKERNAME/EMAIL
-P stratum1+tcp://ETH_WALLET@eth-eu2.nanopool.org:9999/WORKERNAME/EMAIL
-P stratum1+tcp://ETH_WALLET@eth-us-east1.nanopool.org:9999/WORKERNAME/EMAIL
-P stratum1+tcp://ETH_WALLET@eth-us-west1.nanopool.org:9999/WORKERNAME/EMAIL
```

Without email:

```
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth-asia1.nanopool.org:9999
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth-eu1.nanopool.org:9999
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth-eu2.nanopool.org:9999
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth-us-east1.nanopool.org:9999
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eth-us-west1.nanopool.org:9999
```

### nicehash.com

```
-P stratum2+tcp://BTC_WALLET.WORKERNAME@daggerhashimoto.br.nicehash.com:3353
-P stratum2+tcp://BTC_WALLET.WORKERNAME@daggerhashimoto.eu.nicehash.com:3353
-P stratum2+tcp://BTC_WALLET.WORKERNAME@daggerhashimoto.hk.nicehash.com:3353
-P stratum2+tcp://BTC_WALLET.WORKERNAME@daggerhashimoto.in.nicehash.com:3353
-P stratum2+tcp://BTC_WALLET.WORKERNAME@daggerhashimoto.jp.nicehash.com:3353
-P stratum2+tcp://BTC_WALLET.WORKERNAME@daggerhashimoto.usa.nicehash.com:3353
```

### sparkpool.com

```
-P stratum1+tcp://ETH_WALLET.WORKERNAME@cn.sparkpool.com:3333
-P stratum1+tcp://ETH_WALLET.WORKERNAME@eu.sparkpool.com:3333
-P stratum1+tcp://ETH_WALLET.WORKERNAME@jp.sparkpool.com:3333
-P stratum1+tcp://ETH_WALLET.WORKERNAME@kr.sparkpool.com:3333
-P stratum1+tcp://ETH_WALLET.WORKERNAME@na-east.sparkpool.com:3333
-P stratum1+tcp://ETH_WALLET.WORKERNAME@na-west.sparkpool.com:3333
-P stratum1+tcp://ETH_WALLET.WORKERNAME@tw.sparkpool.com:3333
```

### whalesburg.com

```
-P stratum1+tcp://ETH_WALLET.WORKERNAME@proxy.pool.whalesburg.com:8082
```
