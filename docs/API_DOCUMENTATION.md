# Ethminer's API documentation

## Table of Contents

* [Introduction](#introduction)
* [Activation and Security](#activation-and-security)
* [Usage](#usage)
* [List of requests](#list-of-requests)
    * [api_authorize](#api_authorize)
    * [miner_ping](#miner_ping)
    * [miner_getstatdetail](#miner_getstatdetail)
    * [miner_getstat1](#miner_getstat1)
    * [miner_getstathr](#miner_getstathr)
    * [miner_restart](#miner_restart)
    * [miner_reboot](#miner_reboot)
    * [miner_shuffle](#miner_shuffle)
    * [miner_getconnections](#miner_getconnections)
    * [miner_setactiveconnection](#miner_setactiveconnection)
    * [miner_addconnection](#miner_addconnection)
    * [miner_removeconnection](#miner_removeconnection)
    * [miner_getscramblerinfo](#miner_getscramblerinfo)
    * [miner_setscramblerinfo](#miner_setscramblerinfo)
    * [miner_pausegpu](#miner_pausegpu)
    * [miner_setverbosity](#miner_setverbosity)

## Introduction

Ethminer implements an API (Application Programming Interface) interface which allows to monitor/control some of the run-time values endorsed by this miner. The API interface is available under the following circumstances:

* If you're using a binary release downloaded from the [releases](https://github.com/ethereum-mining/ethminer/releases) section of this repository
* If you build the application from source ensuring you add the compilation switch `-D APICORE=ON`

## Activation and Security

Whenever the above depicted conditions are met you can take advantage of the API support by adding the `--api-bind` argument to the command line used to launch ethminer. The format of this argument is `--api-bind address:port` where `nnnn` is any valid TCP port number (1-65535) and is required, and the `address` dictates what ip the api will listen on, and is optional, and defaults to "all ipv4 addresses". Examples:

```shell
./ethminer [...] --api-bind 3333
```

This example puts the API interface listening on port 3333 of **any** local IPv4 address which means the loop-back interface (127.0.0.1/127.0.1.1) and any configured IPv4 address of the network card(s). To only listen to localhost connections (which may be a more secure setting),

```shell
./ethminer [...] --api-bind 127.0.0.1:3333
```
and likewise, to only listen on a specific address, replace `127.0.0.1` accordingly.



The API interface not only offers monitoring queries but also implements some methods which may affect the functioning of the miner. These latter operations are named _write_ actions: if you want to inhibit the invocation of such methods you may want to put the API interface in **read-only** mode which means only query to **get** data will be allowed and no _write_ methods will be allowed. To do this simply add the - (minus) sign in front of the port number thus transforming the port number into a negative number. Example for read-only mode:

```shell
./ethminer [...] --api-bind -3333
```

_Note. The port number in this examples is taken randomly and does not imply a suggested value. You can use any port number you wish while it's not in use by other applications._

To gain further security you may wish to password protect the access to your API interface simply by adding the `--api-password` argument to the command line sequence, followed by the password you wish. Password may be composed by any printable char and **must not** have spaces. Password checking is **case sensitive**. Example for password protected API interface:

```shell
./ethminer [...] --api-bind -3333 --api-password MySuperSecurePassword!!#123456
```

At the time of writing of this document ethminer's API interface does not implement any sort of data encryption over SSL secure channel so **be advised your passwords will be sent as plain text over plain TCP sockets**.

## Usage

Access to API interface is performed through a TCP socket connection to the API endpoint (which is the IP address of the computer running ethminer's API instance at the configured port). For instance if your computer address is 192.168.1.1 and have configured ethminer to run with `--api-bind 3333` your endpoint will be 192.168.1.1:3333.

Messages exchanged through this channel must conform to the [JSON-RPC 2.0 specification](http://www.jsonrpc.org/specification) so basically you will issue **requests** and will get back **responses**. At the time of writing this document do not expect any **notification**. All messages must be line feed terminated.

To quickly test if your ethminer's API instance is working properly you can issue this simple command:

```shell
echo '{"id":0,"jsonrpc":"2.0","method":"miner_ping"}' | netcat 192.168.1.1 3333
```

and will get back a response like this:

```shell
{"id":0,"jsonrpc":"2.0","result":"pong"}
```

This shows the API interface is live and listening on the configured endpoint.

## List of requests

|   Method  | Description  | Write Protected |
| --------- | ------------ | --------------- |
| [api_authorize](#api_authorize) | Issues the password to authenticate the session | No |
| [miner_ping](#miner_ping) | Responds back with a "pong" | No |
| [miner_getstatdetail](#miner_getstatdetail) | Request the retrieval of operational data in most detailed form | No
| [miner_getstat1](#miner_getstat1) | Request the retrieval of operational data in compatible format | No
| [miner_getstathr](#miner_getstathr) | Request the retrieval of operational data in Ethminer's format | No
| [miner_restart](#miner_restart) | Instructs ethminer to stop and restart mining | Yes |
| [miner_reboot](#miner_reboot) | Try to launch reboot.bat (on Windows) or reboot.sh (on Linux) in the ethminer executable directory | Yes
| [miner_shuffle](#miner_shuffle) | Initializes a new random scramble nonce | Yes
| [miner_getconnections](#miner_getconnections) | Returns the list of connections held by ethminer | No
| [miner_setactiveconnection](#miner_setactiveconnection) | Instruct ethminer to immediately connect to the specified connection | Yes
| [miner_addconnection](#miner_addconnection) | Provides ethminer with a new connection to use | Yes
| [miner_removeconnection](#miner_removeconnection) | Removes the given connection from the list of available so it won't be used again | Yes
| [miner_getscramblerinfo](#miner_getscramblerinfo) | Retrieve information about the nonce segments assigned to each GPU | No
| [miner_setscramblerinfo](#miner_setscramblerinfo) | Sets information about the nonce segments assigned to each GPU | Yes
| [miner_pausegpu](#miner_pausegpu) | Pause/Start mining on specific GPU | Yes

### api_authorize

If your API instance is password protected by the usage of `--api-password` any remote trying to interact with the API interface **must** send this method immediately after connection to get authenticated. The message to send is:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "api_authorize",
  "params": {
    "psw": "MySuperSecurePassword!!#123456"
  }
}
```

where the member `psw` **must** contain the very same password configured with `--api-password` argument. As expected result you will get a JSON-RPC 2.0 response with positive or negative values. For instance if the password matches you will get a response like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": true,
}
```

or, in case of any error:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "error": {
    "code": -401,
    "message": "Invalid password"
  }
}
```

### miner_ping

This method is primarily used to check the liveness of the API interface.

To invoke the action:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_ping"
}
```

and expect back a result like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": "pong"
}
```

which confirms the action has been performed.

If you get no response or the socket timeouts it's likely your ethminer's instance has become unresponsive (or in worst cases the OS of your mining rig is unresponsive) and needs to be re-started/re-booted.

### miner_getstatdetail

With this method you expect back a detailed collection of statistical data. To issue a request:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_getstatdetail"
}
```

and expect back a response like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": {
    "connection": {                     // Current active connection
      "isconnected": true,
      "switched": 0,
      "uri": "stratum+tcp://<omitted-ethereum-address>.worker@eu1.ethermine.org:4444"
    },
    "difficulty": 3999938964.0,
    "epoch": 218,
    "epoch_changes": 1,                 // Ethminer starts with epoch 0. First connection to pool increments this counter
    "hashrate": 46709128,               // Overall HashRate in H/s
    "hostname": "<omitted-hostname>",
    "runtime": 4,                       // Total running time in minutes
    "shares": {                         // Summarized info about shares
      "accepted": 5,
      "acceptedstale": 1,
      "invalid": 1,
      "rejected": 0
    },
    "tstart": 63,
    "tstop": 69,
    "version": "ethminer-0.16.0.dev3-73+commit.f35c22ab",
    "gpus": [
      {"fan": 54,                       // Fan in %
       "hashrate": 23604564,            // HashRate of GPU in H/s
       "index": 0,
       "ispaused": false,
       "nonce_start": 6636918940706763208,
       "nonce_stop": 6636920040218390984,
       "pause_reason": "",              // Possible values: "", "temperature", "api", or "temperature,api"
       "power": 0.0,                    // Powerdrain in W
       "shares": {                      // Detailed info about shares from this GPU
         "accepted": 3,
         "acceptedstale": 0,
         "invalid": 0,
         "lastupdate": 1,               // Share info from this GPU updated X minutes ago
         "rejected": 0
       },
       "temp": 53                       // Temperature in °C
      },
      {"fan": 53,
       "hashrate": 23104564,
       "index": 1,
       "ispaused": false,
       "nonce_start": 6636920040218391000,
       "nonce_stop": 6636921139730018000,
       "pause_reason": "",
       "power": 0.0,
       "shares": {
         "accepted": 2,
         "acceptedstale": 1,
         "invalid": 1,
         "lastupdate": 2,
         "rejected": 0
       },
       "temp": 56
      }
    ]
  }
}
```

If values not set (eg --tstart) or the underlaying function returns an error expect `null` as returned value!


### miner_getstat1

With this method you expect back a collection of statistical data. To issue a request:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_getstat1"
}
```

and expect back a response like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": [
    "ethminer-0.16.0.dev0+commit.41639944", // Running ethminer's version
    "48",                                   // Total running time in minutes
    "87221;54;0",                           // ETH hashrate in KH/s, submitted shares, rejected shares
    "14683;14508;14508;14508;14508;14508",  // Detailed ETH hashrate in KH/s per GPU
    "0;0;0",                                // DCR hashrate in KH/s, submitted shares, rejected shares (not used)
    "off;off;off;off;off;off",              // Detailed DCR hashrate in KH/s per GPU (not used)
    "53;90;50;90;56;90;58;90;61;90;60;90",  // Temp and fan speed pairs per GPU
    "eu1.ethermine.org:4444",               // Mining pool currently active
    "0;0;0;0"                               // ETH invalid shares, ETH pool switches, DCR invalid shares, DCR pool switches
  ]
}
```

Some of the arguments here expressed have been set for compatibility with other miners so their values are not set. For instance, ethminer **does not** support dual (ETH/DCR) mining.

### miner_getstathr

With this method you expect back a collection of statistical data. To issue a request:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_getstathr"
}
```

and expect back a response like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": {
    "ethhashrate": 73056881,            // Overall HashRate in H/s
    "ethhashrates": [                   // Hashrate per GPU in H/S
      14681287,
      14506510,
      14681287,
      14506510,
      0,
      14681287
    ],
    "ethinvalid": 0,                    // Total number of invalid shares
    "ethpoolsw": 0,                     // Total number of pool switches
    "ethrejected": 0,                   // Total number of rejected shares
    "ethshares": 64,                    // Total number of found and submitted shares
    "fanpercentages": [                 // Fan percentages per GPU
       90,
       90,
       90,
       90,
       100,
       90
    ],
    "pooladdrs": "eu1.ethermine.org:4444",  // Mining pool currently active
    "powerusages": [                        // Power draw (in W) per GPU
       0.0,
       0.0,
       0.0,
       0.0,
       0.0,
       0.0
    ],
    "runtime": "59",                    // Total runtime in minutes
    "temperatures": [                   // Temperatures per GPU
       53,
       50,
       56,
       58,
       68,
       60
    ],
    "ispaused": [                       // Is mining paused per GPU
       false,
       false,
       false,
       false,
       true,
       false
    ],
    "version": "ethminer-0.16.0.dev0+commit.41639944" // Running ethminer's version
  }
}
```

This format does not honor any compliance with other miners' format and does not express values from dual mining, which, we reiterate, is not supported by ethminer.

### miner_restart

With this method you instruct ethminer to _restart_ mining. Restarting means:

* Stop actual mining work
* Unload generated DAG files
* Reset devices (GPU)
* Regenerate DAG files
* Restart mining

The invocation of this method **_may_** be useful if you detect one or more GPUs are in error, but in a recoverable state (eg. no hashrate but the GPU has not fallen off the bus). In other words, this method works like stopping ethminer and restarting it **but without loosing connection to the pool**.

To invoke the action:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_restart"
}
```

and expect back a result like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": true
}
```

which confirms the action has been performed.

**Note**: This method is not available if the API interface is in read-only mode (see above).

### miner_reboot

With this method you instruct ethminer to execute reboot.bat (on Windows) or reboot.sh (on Linux) script which must exists and being executable in the ethminer directory.
As ethminer has no idea what's going on in the script, ethminer continues with it's normal work.
If you invoke this function `api_miner_reboot` is passed to the script as first parameter.

To invoke the action:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_reboot"
}
```

and expect back a result like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": true
}
```

which confirms an executable file was found and ethminer tried to start it.

**Note**: This method is not available if the API interface is in read-only mode (see above).

### miner_shuffle

The mining process is nothing more that finding the right number (nonce) which, applied to an algorithm (ethash) and some data, gives a result which is below or equal to a given target. This is very very (very) short!
The range of nonces to be searched is a huge number: 2^64 = 18446744073709600000~ possible values. Each one has the same probability to be the _right_ one.

Every time ethminer receives a job from a pool you'd expect the miner to begin searching from the first, but that would be boring. So the concept of scramble nonce has been introduced to achieve these goals:

* Start the searching from a random point within the range
* Ensure all GPUs do not search the same data, or, in other words, ensure each GPU searches its own range of numbers without overlapping with the same numbers of the other GPUs

All `miner_shuffle` method does is to re-initialize a new random scramble nonce to start from in next jobs.

To invoke the action:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_shuffle"
}
```

and expect back a result like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": true
}
```

which confirms the action has been performed.

### miner_getconnections

When you launch ethminer you provide a list of connections specified by the `-P` argument. If you want to remotely check which is the list of connections ethminer is using, you can issue this method:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_getconnections"
}
```

and expect back a result like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": [
    {
      "active": false,
      "index": 0,
      "uri": "stratum+tcp://<omitted-ethereum-address>.worker@eu1.ethermine.org:4444"
    },
    {
      "active": true,
      "index": 1,
      "uri": "stratum+tcp://<omitted-ethereum-address>.worker@eu1.ethermine.org:14444"
    },
    {
      "active": false,
      "index": 2,
      "uri": "stratum+tcp://<omitted-ethereum-classic-address>.worker@eu1-etc.ethermine.org:4444"
    }
  ]
}
```

The `result` member contains an array of objects, each one with the definition of the connection (in the form of the URI entered with the `-P` argument), its ordinal index and the indication if it's the currently active connetion.

### miner_setactiveconnection

Given the example above for the method [miner_getconnections](#miner_getconnections) you see there is only one active connection at a time. If you want to control remotely your mining facility and want to force the switch from one connection to another you can issue this method:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_setactiveconnection",
  "params": {
    "index": 0
  }
}
```

You have to pass the `params` member as an object which has member `index` valued to the ordinal index of the connection you want to activate. As a result you expect one of the following:

* Nothing happens if the provided index is already bound to an _active_ connection
* If the selected index is not of an active connection then ethminer will disconnect from currently active connection and reconnect immediately to the newly selected connection
* An error result if the index is out of bounds or the request is not properly formatted

**Please note** that this method changes the runtime behavior only. If you restart ethminer from a batch file the active connection will become again the first one of the `-P` arguments list.

### miner_addconnection

If you want to remotely add a new connection to the running instance of ethminer you can use this this method by sending a message like this

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_addconnection",
  "params": {
    "uri": "stratum+tcp://<ethaddress>.<workername>@eu1.ethermine.org:4444"
  }
}
```

You have to pass the `params` member as an object which has member `uri` valued exactly the same way you'd add a connection using the `-P` argument. As a result you expect one of the following:

* An error if the uri is not properly formatted
* An error if you try to _mix_ stratum mode with getwork mode (which begins with `http://`)
* A success message if the newly defined connection has been properly added

Eventually you may want to issue [miner_getconnections](#miner_getconnections) method to identify which is the ordinal position assigned to the newly added connection and make use of [miner_setactiveconnection](#miner_setactiveconnection) method to instruct ethminer to use it immediately.

**Please note** that this method changes the runtime behavior only. If you restart ethminer from a batch file the added connection won't be available if not present in the `-P` arguments list.

### miner_removeconnection

Recall once again the example for the method [miner_getconnections](#miner_getconnections). If you wish to remove the third connection (the Ethereum classic one) from the list of connections (so it won't be used in case of failover) you can send this method:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_removeconnection",
  "params": {
    "index": 2
  }
}
```

You have to pass the `params` member as an object which has member `index` valued to the ordinal index (zero based) of the connection you want to remove. As a result you expect one of the following:

* An error if the index is out of bounds **or if the index corresponds to the currently active connection**
* A success message. In such case you can later reissue [miner_getconnections](#miner_getconnections) method to check the connection has been effectively removed.

**Please note** that this method changes the runtime behavior only. If you restart ethminer from a batch file the removed connection will become again again available if provided in the `-P` arguments list.

### miner_getscramblerinfo

When searching for a valid nonce the miner has to find (at least) 1 of possible 2^64 solutions. This would mean that a miner who claims to guarantee to find a solution in the time of 1 block (15 seconds for Ethereum) should produce 1230 PH/s (Peta hashes) which, at the time of writing, is more than 4 thousands times the whole hashing power allocated worldwide for Ethereum.
This gives you an idea of numbers in play. Luckily a couple of factors come in our help: difficulty and time. We can imagine difficulty as a sort of judge who determines how many of those possible solutions are valid. And the block time which allows the miner to stay longer on a sequence of numbers to find the solution.
This all said it's however impossible for any miner (no matter if CPU or GPU or even ASIC) to cover the most part of this huge range in reasonable amount of time. So we need to resign to examine and test only a small fraction of this range.

Ethminer, at start, randomly chooses a scramble_nonce, a random number picked in the 2^64 range to start checking nonces from. In addition ethminer gives each GPU a unique, non overlapping, range of nonces called _segment_. Segments ensure no GPU does the same job of another GPU thus avoiding two GPU find the same result.
To accomplish this each segment has a range 2^40 nonces by default. If you want to check which is the scramble_nonce and which are the segments assigned to each GPU you can issue this method:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_getscramblerinfo"
}
```

and expect a result like this:

```js
{
  "id": 0,
  "jsonrpc": "2.0",
  "result": {
    "noncescrambler": 16704043538687679721,
    "segments": [
      {
        "gpu": 0,
        "start": 16704043538687679721,
        "stop": 16704044638199307497
      },
      {
        "gpu": 1,
        "start": 16704044638199307497,
        "stop": 16704046837222563049
      },
      {
        "gpu": 2,
        "start": 16704045737710935273,
        "stop": 16704049036245818601
      },
      {
        "gpu": 3,
        "start": 16704046837222563049,
        "stop": 16704051235269074153
      },
      {
        "gpu": 4,
        "start": 16704047936734190825,
        "stop": 16704053434292329705
      },
      {
        "gpu": 5,
        "start": 16704049036245818601,
        "stop": 16704055633315585257
      }
    ],
    "segmentwidth": 40
  }
}
```

Note that segment width is the exponent in the expression `pow(2, segment)`.
The information hereby exposed may be used in large mining operations to check whether or not two (or more) rigs may result having overlapping segments. The possibility is very remote ... but is there.

### miner_setscramblerinfo

To approach this method you have to read carefully the method [miner_getscrambleinfo](#miner_getscrambleinfo) and what it reports. By the use of this method you can set a new scramble_nonce and/or set a new segment width:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_setscramblerinfo",
  "params": {
    "noncescrambler": 16704043538687679721,      // At least one of these two members
    "segmentwidth": 38                           // or both.
  }
}
```

This will adjust nonce scrambler and segment width assigned to each GPU. This method is intended only for highly skilled people who do a great job in math to determine the optimal values for large mining operations.
**Use at your own risk**

### miner_pausegpu

Pause or (restart) mining on specific GPU.
This ONLY (re)starts mining if GPU was paused via a previous API call and not if GPU pauses for other reasons.

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_pausegpu",
  "params": {
    "index": 0,
    "pause": true
  }
}
```

and expect a result like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": true
}
```

which confirms the action has been performed.
Again: This ONLY (re)starts mining if GPU was paused via a previous API call and not if GPU pauses for other reasons.

### miner_setverbosity

Set the verbosity level of ethminer.

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_setverbosity",
  "params": {
    "verbosity": 9
  }
}
```

and expect a result like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": true
}
```
