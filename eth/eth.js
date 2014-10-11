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
            // properties
            { "method": "coinbase", "params": null, "order": [], "returns" : "" },
            { "method": "isListening", "params": null, "order": [], "returns" : false },
            { "method": "setListening", "params": { "l": "" }, "order" : ["l"], "returns" : ""},
            { "method": "isMining", "params": null, "order": [], "returns" : false },
            { "method": "setMining", "params": { "l": "" }, "order" : ["l"], "returns" : ""},
            { "method": "gasPrice", "params": null, "order": [], "returns" : "" },
            { "method": "key", "params": null, "order": [], "returns" : "" },
            { "method": "keys", "params": null, "order": [], "returns" : [] },
            { "method": "peerCount", "params": null, "order": [], "returns" : 0 },
            { "method": "defaultBlock", "params": null, "order": [], "returns" : 0},
            { "method": "number", "params": null, "order": [], "returns" : 0},

            // synchronous getters
            { "method": "balanceAt", "params": { "a": "", "block": 0}, "order": ["a", "block"], "returns" : ""},
            { "method": "stateAt", "params": { "a": "", "p": "", "block": 0}, "order": ["a", "p", "block"], "returns": ""},
            { "method": "countAt", "params": { "a": "", "block": 0}, "order": ["a", "block"], "returns" : ""},
            { "method": "codeAt", "params": { "a": "", "block": 0}, "order": ["a", "block"], "returns": ""},

            // transactions
            { "method": "transact", "params": { "json": ""}, "order": ["json"], "returns": ""},
            { "method": "call", "params": { "json": []}, "order": ["json"], "returns": ""},

            // blockchain
            { "method": "block", "params": { "numberOrHash": ""}, "order": ["numberOrHash"], "returns": []},
            { "method": "transaction", "params": { "numberOrHash": "", "i": ""}, "order": ["numberOrHash", "i"], "returns": ""},
            { "method": "uncle", "params": { "numberOrHash": "", "i": ""}, "order": ["numberOrHash", "i"], "returns": ""},

            // watches and message filtering
            { "method": "messages", "params": { "json": ""}, "order": ["json"], "returns": ""},
            { "method": "watch", "params": { "json": ""}, "order": ["json"], "returns": ""},

            // misc
            { "method": "secretToAddress", "params": { "s": ""}, "order": ["s"], "returns": ""},
            { "method": "lll", "params": { "s": ""}, "order": ["s"], "returns": ""},
            { "method": "sha3", "params": { "s": ""}, "order": ["s"], "returns": ""},   // TODO other sha3
            { "method": "toAscii", "params": { "s": ""}, "order": ["s"], "returns": ""},
            { "method": "fromAscii", "params": { "s": "", "padding": 0}, "order": ["s", "padding"], "returns": ""}, 
            { "method": "toDecimal", "params": {"s": ""}, "order": ["s"], "returns" : ""},
            { "method": "toFixed", "params": {"s": ""}, "order": ["s"], "returns" : ""},
            { "method": "fromFixed", "params": {"s": ""}, "order": ["s"], "returns" : 0.0},
            { "method": "offset", "params": {"s": "", "offset": ""}, "order": ["s", "offset"], "returns" : ""},
];

window.eth = (function ethScope() {
	var m_reqId = 0
	var ret = {}
    function reformat(m, d) { return m == "lll" ? d.bin() : d; }
	function reqSync(m, p) {
		var req = { "jsonrpc": "2.0", "method": m, "params": p, "id": m_reqId }
		m_reqId++
		var request = new XMLHttpRequest();	
        request.open("POST", "http://localhost:8080", false)
//		console.log("Sending " + JSON.stringify(req))
        request.send(JSON.stringify(req))
        return reformat(m, JSON.parse(request.responseText).result)
	}
	function reqAsync(m, p, f) {
		var req = { "jsonrpc": "2.0", "method": m, "params": p, "id": m_reqId }
		m_reqId++
		var request = new XMLHttpRequest();	
        request.open("POST", "http://localhost:8080", true)
        request.send(JSON.stringify(req))
		request.onreadystatechange = function() {
			if (request.readyState === 4)
                f(reformat(m, JSON.parse(request.responseText).result))
	    };
	}
	function isEmpty(obj) {
		for (var prop in obj)
		    if (obj.hasOwnProperty(prop))
		        return false
		return true
	}

	var m_watching = {};
	
	for (si in spec) (function(s) {
		var m = s.method;
		var am = "get" + m.slice(0, 1).toUpperCase() + m.slice(1);
		var getParams = function(a) {
			var p = s.params ? {} : null;
			if (m == "stateAt")
				if (a.length == 2)
					a[2] = "0";
				else
					a[2] = String(a[2]);
			for (j in s.order)
				p[s.order[j]] = (s.order[j][0] === "b") ? a[j].unbin() : a[j];
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
	return ret;
}());

}

