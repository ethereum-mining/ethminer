var prettyPrint = (function () {
    var onlyDecentPlatform = function (x) {
        return env.os.indexOf('Windows') === -1 ? x : '';
    };

    var color_red = onlyDecentPlatform('\033[31m');
    var color_green = onlyDecentPlatform('\033[32m');
    var color_pink = onlyDecentPlatform('\033[35m');
    var color_white = onlyDecentPlatform('\033[0m');
    var color_blue = onlyDecentPlatform('\033[30m');

    function pp(object, indent) {
        try {
            JSON.stringify(object)
        } catch(e) {
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
            str += "]";
        } else if (object instanceof Error) {
            str += color_red + "Error: " + color_white + object.message;
        }  else if (object === null) {
            str += color_blue + "null";
        } else if(typeof(object) === "undefined") {
            str += color_blue + object;
        } else if (isBigNumber(object)) {
            str += color_green + object.toString(10) + "'";
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
            str += color_green + "'" + object + "'";
        } else if(typeof(object) === "number") {
            str += color_red + object;
        } else if(typeof(object) === "function") {
            str += color_pink + "[Function]";
        } else {
            str += object;
        }
        str += color_white;
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
        return (!!object.constructor && object.constructor.name === 'BigNumber') ||
            (typeof BigNumber !== 'undefined' && object instanceof BigNumber)
    };
    function prettyPrintInner(/* */) {
        var args = arguments;
        var ret = "";
        for(var i = 0, l = args.length; i < l; i++) {
           ret += pp(args[i], "");
        }
        return ret;
    };
    return prettyPrintInner;
})();
