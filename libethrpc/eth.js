// https://github.com/ethereum/cpp-ethereum/wiki/PoC-6-JS-API

if (typeof(window.eth) === "undefined")
{
if (typeof(require) !== "undefined")
	require( ['ethString'], function() {} )
else if (typeof(String.prototype.pad) === "undefined")
{
	var scriptTag = document.getElementsByTagName('script');
	scriptTag = scriptTag[scriptTag.length - 1]; 
	var scriptPath = scriptTag.src; 
	var path = scriptPath.substr(0, scriptPath.lastIndexOf( '/' ));
	var start = '<script src="' + path + '/';
	var slash = '"><'+'/script>';
	document.write(start + 'BigInteger.js' + slash);
	document.write(start + 'ethString.js' + slash);
}

var spec = [
            { "method": "coinbase", "params": null, "order": [], "returns" : "" },
            { "method": "setCoinbase", "params": { "address": "" }, "order": ["address"], "returns" : true },
            { "method": "listening", "params": null, "order": [], "returns" : false },
            { "method": "setListening", "params": { "listening": false }, "order" : ["listening"], "returns" : true },
            { "method": "mining", "params": null, "order": [], "returns" : false },
            { "method": "setMining", "params": { "mining": false }, "order" : ["mining"], "returns" : true },
            { "method": "gasPrice", "params": null, "order": [], "returns" : "" },
            { "method": "key", "params": null, "order": [], "returns" : "" },
            { "method": "keys", "params": null, "order": [], "returns" : [] },
            { "method": "peerCount", "params": null, "order": [], "returns" : 0 },
            { "method": "defaultBlock", "params": null, "order": [], "returns" : 0},
            { "method": "number", "params": null, "order": [], "returns" : 0},

            { "method": "balanceAt", "params": { "address": "", "block": 0}, "order": ["address", "block"], "returns" : ""},
            { "method": "stateAt", "params": { "address": "", "storage": "", "block": 0}, "order": ["address", "storage", "block"], "returns": ""},
            { "method": "countAt", "params": { "address": "", "block": 0}, "order": ["address", "block"], "returns" : 0.0},
            { "method": "codeAt", "params": { "address": "", "block": 0}, "order": ["address", "block"], "returns": ""},

            { "method": "transact", "params": { "json": {}}, "order": ["json"], "returns": ""},
            { "method": "call", "params": { "json": {}}, "order": ["json"], "returns": ""},

            { "method": "block", "params": { "params": {}}, "order": ["params"], "returns": {}},
            { "method": "transaction", "params": { "params": {}, "i": 0}, "order": ["params", "i"], "returns": {}},
            { "method": "uncle", "params": { "params": {}, "i": 0}, "order": ["params", "i"], "returns": {}},

            { "method": "messages", "params": { "params": {}}, "order": ["params"], "returns": []},
            { "method": "watch", "params": { "params": ""}, "order": ["params"], "returns": 0},
            { "method": "check", "params": { "id": 0}, "order": [], "returns": true},
            { "method": "killWatch", "params": { "id": 0}, "order": ["params"], "returns": true},

            { "method": "secretToAddress", "params": { "s": ""}, "order": ["s"], "returns": ""},
            { "method": "lll", "params": { "s": ""}, "order": ["s"], "returns": ""},
            { "method": "sha3", "params": { "s": ""}, "order": ["s"], "returns": ""},
            { "method": "toAscii", "params": { "s": ""}, "order": ["s"], "returns": ""},
            { "method": "fromAscii", "params": { "s": "", "padding": 0}, "order": ["s", "padding"], "returns": ""}, 
            { "method": "toDecimal", "params": {"s": ""}, "order": ["s"], "returns" : ""},
            { "method": "toFixed", "params": {"s": 0.0}, "order": ["s"], "returns" : ""},
            { "method": "fromFixed", "params": {"s": ""}, "order": ["s"], "returns" : 0.0}
];

window.eth = (function ethScope() {
	var m_reqId = 0
	var ret = {}
    function reformat(m, d) { return m == "lll" ? d.bin() : d; };
	function reqSync(m, p) {
		var req = { "jsonrpc": "2.0", "method": m, "params": p, "id": m_reqId }
		m_reqId++
		var request = new XMLHttpRequest();	
        request.open("POST", "http://localhost:8080", false)
        request.send(JSON.stringify(req))
        return reformat(m, JSON.parse(request.responseText).result)
	};
	function reqAsync(m, p, f) {
		var req = { "jsonrpc": "2.0", "method": m, "params": p, "id": m_reqId }
		m_reqId++
		var request = new XMLHttpRequest();	
        request.open("POST", "http://localhost:8080", true)
        request.send(JSON.stringify(req))
		request.onreadystatechange = function() {
			if (request.readyState === 4)
                if (f)
                    f(reformat(m, JSON.parse(request.responseText).result));
	    };
	};
    
    var getParams = function (spec, name, args) {
        var setup = spec.filter(function (s) {
            return s.method === name;
        });

        if (setup.length === 0) {
            return {};
        }

        var paramSetup = setup[0];

        var p = paramSetup.params ? {} : null;
        for (j in paramSetup.order)
            p[paramSetup.order[j]] = args[j];
        return p;
    };

    var addPrefix = function (s, prefix) {
        if (!s) {
            return s;
        }
        return prefix + s.slice(0, 1).toUpperCase() + s.slice(1);
    };

    var toGetter = function (s) {
        return addPrefix(s, "get");
    };

    var toSetter = function (s) {
        return addPrefix(s, "set");
    };

    var defaults = function (def, obj) {
        if (!def) {
            return obj;
        }
        var rewriteProperties = function (dst, p) {
            Object.keys(p).forEach(function (key) {
                if (p[key] !== undefined) {
                    dst[key] = p[key];
                }
            });
        };
        var res = {};
        rewriteProperties(res, def);
        rewriteProperties(res, obj);
        return res;
    };

    var setupProperties = function (root, spec) {
        var properties = [
        { name: "coinbase", getter: "coinbase", setter: "setCoinbase" },
        { name: "listening", getter: "listening", setter: "setListening" },
        { name: "mining", getter: "mining", setter: "setMining" },
        { name: "gasPrice", getter: "gasPrice"},
        { name: "key", getter: "key" },
        { name: "keys", getter: "keys" },
        { name: "peerCount", getter: "peerCount" },
        { name: "defaultBlock", getter: "defaultBlock" },
        { name: "number", getter: "number" }];
        
        properties.forEach(function (property) {
            var p = {};
            if (property.getter) {
                p.get = function () {
                    return reqSync(property.getter, {});
                };
                root[toGetter(property.name)] = function (f) {
                    return reqAsync(property.getter, null, f);
                };
            }
            if (property.setter) {
                p.set = function (newVal) {
                    return reqSync(property.setter, getParams(spec, property.setter, arguments));
                };
                root[toSetter(property.name)] = function (newVal, f) {
                    return reqAsync(property.setter, getParams(spec, property.setter, arguments), f);
                };
            }

            Object.defineProperty(root, property.name, p);
        });
    };

    var setupMethods = function (root, spec) {
        var methods = [
        { name: "balanceAt", async: "getBalanceAt", default: {block: 0} },
        { name: "stateAt", async: "getStateAt", default: {block: 0} },
        { name: "countAt", async: "getCountAt", default: {block: 0} },
        { name: "codeAt", async: "getCodeAt", default: {block: 0} },
        { name: "transact", async: "makeTransact"},
        { name: "call", async: "makeCall" },
        { name: "messages", async: "getMessages" },
        { name: "block", async: "getBlock" },
        { name: "transaction", async: "getTransaction" },
        { name: "uncle", async: "getUncle" }
        ];

        methods.forEach(function (method) {
            root[method.name] = function () {
                return reqSync(method.name, defaults(method.default, getParams(spec, method.name, arguments)));
            };
            if (method.async) {
                root[method.async] = function () {
                    return reqAsync(method.name, defaults(method.default, getParams(spec, method.name, arguments)), arguments[arguments.length - 1]);
                };
            };
        });
    };

    var setupWatch = function (root) {
        root.watch = function (val) {
            if (typeof val !== 'string') {
                val = JSON.stringify(val);
            }
    
            var id;
            reqAsync('watch', {params: val}, function (result) {
                id = result;
            }); // async send watch
            var callbacks = [];
            var exist = true;
            var w = {
                changed: function (f) {
                    callbacks.push(f);
                },
                uninstall: function (f) {
                    reqAsync('killWatch', {id: id}); 
                    exist = false;
                },
                messages: function () {
                    // TODO! 
                },
                getMessages: function (f) {
                    // TODO!
                }
            };

            var check = function () {
                if (!exist) {
                    return;
                }
                if (callbacks.length) {
                    reqAsync('check', {id: id}, function (res) { 
                        if (!res) {
                            return;
                        }
                        callbacks.forEach(function (f) {
                            f();
                        });
                    });
                }
                window.setTimeout(check, 12000);
            };

            check();
            return w;
        }
    };

    setupProperties(ret, window.spec);
    setupMethods(ret, window.spec);
    setupWatch(ret);

	return ret;
}());

}

