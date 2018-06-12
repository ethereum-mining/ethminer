# Ethminer's API documentation

Ethminer offers implements an API (Application Programming Interface) interface which allows to monitor/control some of the run-time values endorsed by this miner. The API interface is available under the following circumstances:

* If you're using a binary release downloaded from the [releases](https://github.com/ethereum-mining/ethminer/releases) section of this repository
* If you build the application from source ensuring you add the compilation switch `-D APICORE=ON`

## Activation and Security

Whenever the above depicted conditions are met you can take advantage of the API support by adding the `--api-port` argument to the command line used to launch ethminer. The format of this argument is `--api-port nnnn` where `nnnn` is any valid TCP port number (1-65535). Examples:

```
./ethminer [...] --api-port 3333
```

This example puts the API interface listening on port 3333 of **any** local IP address which means the loop-back interface (127.0.0.1/127.0.1.1) and any configured IP address of the network card.

The API interface not only offers monitoring queries but also implements some methods which may affect the functioning of the miner. These latter operations are named _write_ actions: if you want to inhibit the invocation of such methods you may want to put the API interface in **read-only** mode which means only query to **get** data will be allowed and no _write_ methods will be allowed. To do this simply add the - (minus) sign in front of the port number thus transforming the port number into a negative number. Example for read-only mode:

```
./ethminer [...] --api-port -3333
```  

_Note. The port number in this examples is taken randomly and does not imply a suggested value. You can use any port number you wish while it's not in use by other applications._

To gain further security you may wish to password protect the access to your API interface simply by adding the `--api-password` argument to the command line sequence, followed by the password you wish. Password may be composed by any printable char and **must not** have spaces. Password checking is **case sensitive**. Example for password protected API interface:

```
./ethminer [...] --api-port -3333 --api-password MySuperSecurePassword!!#123456
```  

At the time of writing of this document ethminer's API interface does not implement any sort of data encryption over SSL secure channel so **be advised your passwords will be sent as plain text over plain TCP sockets**.

## Usage

Access to API interface is performed through a TCP socket connection to the API endpoint (which is the IP address of the computer running ethminer's API instance at the configured port). For instance if your computer address is 192.168.1.1 and have configured ethminer to run with `--api-port 3333` your endpoint will be 192.168.1.1:3333.

Messages exchanged through this channel must conform the [Jsonrpc v.2 specifications](http://www.jsonrpc.org/specification) so basically you will issue **requests** and will get back **responses**. At the time of writing of this document do not expect any **notification**. All messages must be line feed terminated.

To quick test if your ethminer's API instance is working properly you can issue this simple command

```
echo '{"id":0,"jsonrpc":"2.0","method":"miner_ping"}' | netcat 192.168.1.1 3333
```

and will get back a response like this

```
{"id":0,"jsonrpc":"2.0","result":"pong"}
```

This shows the API interface is live and listening on the configured endpoint.

## List of requests

|   Method  | Description  | Write Protected |
| --------- | ------------ | --------------- |
| [api_authorize](#api_authorize) | Issues the password to authenticate the session | No |
| [miner_getstat1](#miner_getstat1) | Request the retrieval of operational data in compatible format | No
| [miner_getstathr](#miner_getstathr) | Request the retrieval of operational data in Ethminer's format | No
| [miner_ping](#miner_ping) | Responds back with a "pong" | No |
| [miner_restart](#miner_restart) | Instructs ethminer to stop and restart mining | Yes |
| miner_reboot | Not yet implemented | Yes
| [miner_shuffle](#miner_shuffle) | Initializes a new random scramble nonce | Yes

### api_authorize

If your API instance is password protected by the usage of `--api-password` any remote trying to interact with the API interface **must** send this method immediately after connection to get authenticated. The message to send is:

```
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "api_authorize",
  "params": {
    "psw": "MySuperSecurePassword!!#123456"
  }
}
```

where the member `psw` **must** contain the very same password configured with `--api-password` argument. As expected result you will get a JsonRpc 2.0 response with positive or negative values. For instance if password do match you will get a response like this

```
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": true,
}
```

or, in case of any error

```
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

```
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_ping"
}
```

and expect back a result like this

```
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": "pong"
}
```

which confirms the action has been performed.

If you get no response or the socket goes timeout it likely your ethminer's instance has become unresponsive (or in worst cases all the OS of your mining rig is unresponsive) and need to be re-started/re-booted.

### miner_getstat1

With this method you expect back a collection of statistical data. To issue request:

```
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_getstat1"
}
```

and expect back a response like this

```
{  
   "id":1,
   "jsonrpc":"2.0",
   "result":[  
    "0.16.0.dev0+commit.41639944",          // The actual release of ethminer
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

With this method you expect back a collection of statistical data. To issue request:

```
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_getstathr"
}
```

and expect back a response like this

```
{  
   "id":1,
   "jsonrpc":"2.0",
   "result":{  
    "ethhashrate":87563392,             // Overall HashRate in H/s
    "ethhashrates":[                    // Hashrate per GPU in H/S
       14681287,
       14506510,
       14681287,
       14506510,
       14506510,
       14681287
    ],
    "ethinvalid":0,                     // Total number of invalid shares
    "ethpoolsw":0,                      // Total number of pool switches
    "ethrejected":0,                    // Total number of rejected shares
    "ethshares":64,                     // Total number of found and submitted shares
    "fanpercentages":[                  // Fan percentages per GPU
       90,
       90,
       90,
       90,
       90,
       90
    ],
    "pooladdrs":"eu1.ethermine.org:4444",   // Mining pool currently active
    "powerusages":[                         // Power draw (in W) per GPU
       0.0,
       0.0,
       0.0,
       0.0,
       0.0,
       0.0
    ],
    "runtime":"59",                     // Total runtime in minutes
    "temperatures":[                    // Temperatures per GPU  
       53,
       50,
       56,
       58,
       61,
       60
    ],
    "version":"0.16.0.dev0+commit.41639944" // Running ethminer's version
   }
}
```

This format does not honor any compliance with other miner's format and does not express values from dual mining, which, we reiterate, is not supported by ethminer.

### miner_restart

With this method you instruct ethminer to _restart_ mining. Restarting means:

* Stop actual mining work
* Unload generated DAG files
* Reset devices (GPU)
* Regenerate DAG files
* Restart mining

The invocation of this method **_may_** be useful if you detect one, or more, GPU are in error but in a recoverable state (eg. no hashrate but the GPU has not fallen off the bus). In other words this method works like stopping ethminer and restarting it **but without loosing connection to the pool**.

To invoke the action:

```
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_restart"
}
```

and expect back a result like this

```
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": true
}
```

which confirms the action has been performed.

**Note** This method is not available if API interface is in read-only mode (see above)

### miner_shuffle

The mining process is nothing more that finding the right number (nonce) which, applied to an algorithm (ethash) and some data, gives a result which is below or equal to a given target. This in very very (very) short !
The range of nonces to be searched is a huge number: 2^64 = 18446744073709600000~ possible values. Each one has the same probability to be the _right_ one.

Every time ethminer receives a job from a pool you'd expect the miner to begin searching from the first but that would be boring. So the concept of scramble nonce has been introduced to achieve these goals:

* Start the searching from a random point within the range
* Ensure all GPUs do not search the same data, or, in other words, ensure each GPU searches it's own range of numbers without overlapping with the same numbers of other GPUs

All miner_shuffle method does is to re-initialize a new random scramble nonce to start from in next jobs.

To invoke the action:

```
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_shuffle"
}
```

and expect back a result like this

```
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": true
}
```

which confirms the action has been performed.
