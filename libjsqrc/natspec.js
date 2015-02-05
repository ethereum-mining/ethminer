
/**
 * This plugin exposes 'evaluateExpression' method which should be used
 * to evaluate natspec description
 * It should be reloaded each time we want to evaluate set of expressions
 * Just because of security reasons
 * TODO: make use of sync api (once it's finished) and remove unnecessary 
 * code from 'getContractMethods'
 * TODO: unify method signature creation with abi.js (and make a separate method from it)
 */

/// Should be called to copy values from object to global context
var copyToContext = function (obj, context) {
    var keys = Object.keys(obj);
    keys.forEach(function (key) {
        context[key] = obj[key];
    });
}

/// Function called to get all contract's storage values
/// @returns hashmap with contract properties which are used
var getContractProperties = function (address, abi) {
    return {};
};

/// Function called to get all contract's methods
/// @returns hashmap with used contract's methods
var getContractMethods = function (address, abi) {
    return web3.eth.contract(address, abi);
};

var getMethodWithName = function(abi, name) {
    for (var i = 0; i < abi.length; i++) {
        if (abi[i].name === name) {
            return abi[i];
        }
    }
    console.warn('could not find method with name: ' + name);
    return undefined;
};

/// Function called to get all contract method input variables
/// @returns hashmap with all contract's method input variables
var getContractInputParams = function (abi, methodName, params) {
    var method = getMethodWithName(abi, methodName);
    return method.inputs.reduce(function (acc, current, index) {
        acc[current.name] = params[index];
        return acc;
    }, {});
};

/// Should be called to evaluate single expression
/// Is internally using javascript's 'eval' method
/// Should be checked if it is safe
var evaluateExpression = function (expression) {

    var self = this;
    var abi = web3._currentContractAbi;
    var address = web3._currentContractAddress;
    var methodName = web3._currentContractMethodName;
    var params = web3._currentContractMethodParams;

    var storage = getContractProperties(address, abi); 
    var methods = getContractMethods(address, abi);
    var inputParams = getContractInputParams(abi, methodName, params);

    copyToContext(storage, self);
    copyToContext(methods, self);
    copyToContext(inputParams, self);

    // TODO: test if it is safe
    var evaluatedExpression = "";

    // match everything in `` quotes
    var pattern = /\`(?:\\.|[^`\\])*\`/gim
    var match;
    var lastIndex = 0;
    while ((match = pattern.exec(expression)) !== null) {
        var startIndex = pattern.lastIndex - match[0].length;

        var toEval = match[0].slice(1, match[0].length - 1);

        evaluatedExpression += expression.slice(lastIndex, startIndex);
        evaluatedExpression += eval(toEval).toString();
    
        lastIndex = pattern.lastIndex;
    }

    evaluatedExpression += expression.slice(lastIndex);
    
    return evaluatedExpression;
};

