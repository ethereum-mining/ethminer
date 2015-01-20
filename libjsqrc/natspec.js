
/**
 * This plugin should be reloaded each time we want to evaluate set of expressions
 * Just because of security reasons
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
    var keys = ['test'];

    return keys.reduce(function (acc, current) {
        acc[current] = natspec.stateAt(current);
        return acc;
    }, {});
};

/// Function called to get all contract's methods
/// @returns hashmap with used contract's methods
var getContractMethods = function (expression, abi) {
    var keys = ['testMethod'];

    return keys.reduce(function (acc, current) {
        acc[current] = function () {
            // TODO: connect parser
        };
        return acc;
    }, {});
};

/// Should be called to evaluate single expression
/// Is internally using javascript's 'eval' method
/// Should be checked if it is safe
var evaluateExpression = function (expression) {

    var self = this;
    var abi = web3._currentAbi;

    var storage = getContractProperties(expression, abi); 
    var methods = getContractMethods(expression, abi);

    copyToContext(storage, self);
    copyToContext(methods, self);

    // TODO: check if it is safe
    return eval(expression).toString();
};

