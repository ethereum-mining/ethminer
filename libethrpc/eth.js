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

            { "method": "block", "params": { "numberOrHash": ""}, "order": ["numberOrHash"], "returns": {}},
            { "method": "transaction", "params": { "numberOrHash": "", "i": 0}, "order": ["numberOrHash", "i"], "returns": {}},
            { "method": "uncle", "params": { "numberOrHash": "", "i": 0}, "order": ["numberOrHash", "i"], "returns": {}},

            { "method": "messages", "params": { "json": {}}, "order": ["json"], "returns": []},
            { "method": "watch", "params": { "json": ""}, "order": ["json"], "returns": ""},

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
        { name: "balanceAt", async: "getBalanceAt" },
        { name: "stateAt", async: "getStateAt" },
        { name: "countAt", async: "getCountAt" },
        { name: "codeAt", async: "getCodeAt" },
        { name: "transact", async: "makeTransact" },
        { name: "call", async: "makeCall" },
        { name: "messages", async: "getMessages" },
        { name: "transaction", async: "getTransaction" }
        ];

        methods.forEach(function (method) {
            root[method.name] = function () {
                return reqSync(method.name, getParams(spec, method.name, arguments));
            };
            if (method.async) {
                root[method.async] = function () {
                    return reqAsync(method.name, getParams(spec, method.name, arguments), arguments[arguments.length - 1]);
                };
            };
        });
    };

    setupProperties(ret, window.spec);
    setupMethods(ret, window.spec);

    /*

	function isEmpty(obj) {
		for (var prop in obj)
		    if (obj.hasOwnProperty(prop))
		        return false
		return true
	};

    
	var m_watching = {};
	
	for (si in spec) (function(s) {
		var m = s.method;
		var am = "get" + m.slice(0, 1).toUpperCase() + m.slice(1);
		var getParams = function(a) {
            var p = s.params ? {} : null;
            for (j in s.order)
                p[s.order[j]] = a[j];
			return p
		};
		if (m == "create" || m == "transact")
			ret[m] = function() { return reqAsync(m, getParams(arguments), arguments[s.order.length]) }
		else
		{
			ret[am] = function() { return reqAsync(m, getParams(arguments), arguments[s.order.length]) }
			if (s.params)
				ret[m] = function() { return reqSync(m, getParams(arguments)) }
			else
				Object.defineProperty(ret, m, {
					get: function() { return reqSync(m, {}); },
					set: function(v) {}
				})
		}
	})(spec[si]);

    
	ret.check = function(force) {
		if (!force && isEmpty(m_watching))
			return
		var watching = [];
		for (var w in m_watching)
			watching.push(w)
		var changed = reqSync("check", { "a": watching } );
//		console.log("Got " + JSON.stringify(changed));
		for (var c in changed)
			m_watching[changed[c]]()
		var that = this;
		setTimeout(function() { that.check() }, 12000)
	}

	ret.watch = function(a, fx, f) {
		var old = isEmpty(m_watching)
		if (f)
			m_watching[a + fx] = f
		else
			m_watching[a] = fx
		(f ? f : fx)()
		if (isEmpty(m_watching) != old)
			this.check()
	}
	ret.unwatch = function(f, fx) {
		delete m_watching[fx ? f + fx : f];
	}
	ret.newBlock = function(f) {
		var old = isEmpty(m_watching)
		m_watching[""] = f
		f()
		if (isEmpty(m_watching) != old)
			this.check()
	}
    */
	return ret;
}());

}

