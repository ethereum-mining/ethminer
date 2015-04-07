var prettyPrint = (function () {
    function pp(object, indent) {
        try {
            JSON.stringify(object, null, 2); 
        } catch (e) {
            return pp(e, indent);
        }

        var str = "";
        if(object instanceof Array) {
            str += "[";
            for(var i = 0, l = object.length; i < l; i++) {
                str += pp(object[i], indent);
                if(i < l-1) {
                    str += ", ";
                }
            }
            str += " ]";
        } else if (object instanceof Error) {
            str += "\e[31m" + "Error:\e[0m " + object.message; 
        } else if (isBigNumber(object)) {
            str += "\e[32m'" + object.toString(10) + "'";
        } else if(typeof(object) === "object") {
            str += "{\n";
            indent += "  ";
            var last = getFields(object).pop()
            getFields(object).forEach(function (k) {
                str += indent + k + ": ";
                try {
                    str += pp(object[k], indent);
                } catch (e) {
                    str += pp(e, indent);
                }
                if(k !== last) {
                    str += ",";
                }
                str += "\n";
            });
            str += indent.substr(2, indent.length) + "}";
        } else if(typeof(object) === "string") {
            str += "\e[32m'" + object + "'"; 
        } else if(typeof(object) === "undefined") {
            str += "\e[1m\e[30m" + object;
        } else if(typeof(object) === "number") {
            str += "\e[31m" + object;
        } else if(typeof(object) === "function") {
            str += "\e[35m[Function]";
        } else {
            str += object;
        }
        str += "\e[0m";
        return str;
    }
    var redundantFields = [
        'valueOf',
        'toString',
        'toLocaleString',
        'hasOwnProperty',
        'isPrototypeOf',
        'propertyIsEnumerable',
        'constructor',
        '__defineGetter__',
        '__defineSetter__',
        '__lookupGetter__',
        '__lookupSetter__',
        '__proto__'
    ];
    var getFields = function (object) {
        var result = Object.getOwnPropertyNames(object);
        if (object.constructor && object.constructor.prototype) {
            result = result.concat(Object.getOwnPropertyNames(object.constructor.prototype));
        }
        return result.filter(function (field) {
            return redundantFields.indexOf(field) === -1;
        });
    };
    var isBigNumber = function (object) {
        return typeof BigNumber !== 'undefined' && object instanceof BigNumber;
    };
    function prettyPrintI(/* */) {
        var args = arguments;
        var ret = "";
        for (var i = 0, l = args.length; i < l; i++) {
    	    ret += pp(args[i], "") + "\n";
        }
        return ret;
    }
    return prettyPrintI;
})();

