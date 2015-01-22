
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
var getContractProperties = function (expression, abi) {
    return {};
};

/// Function called to get all contract's methods
/// @returns hashmap with used contract's methods
var getContractMethods = function (address, abi) {
    return web3.eth.contract(address, abi);
};

/// Should be called to evaluate single expression
/// Is internally using javascript's 'eval' method
/// Should be checked if it is safe
var evaluateExpression = function (expression) {

    var self = this;
    var abi = web3._currentContractAbi;
    var address = web3._currentContractAddress;

    var storage = getContractProperties(expression, abi); 
    var methods = getContractMethods(address, abi);

    copyToContext(storage, self);
    copyToContext(methods, self);

    // TODO: test if it is safe
    return eval(expression).toString();
};

