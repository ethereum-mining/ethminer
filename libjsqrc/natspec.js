
/**
 * This plugin should be reloaded each time we want to evaluate set of expressions
 * Just because of security reasons
 */

/// helper variable used by 'copyToGlobalContext' function to copy everything to global context
var _eth_global = this;

/// Should be called to copy values from object to global context
var copyToGlobalContext = function (obj) {
    var keys = Object.keys(obj);
    keys.forEach(function (key) {
        _eth_global[key] = obj[key];
    });
}

/// Function called to get all contract's storage values
/// In future can be improved be getting storage values on demand
/// @returns hashmap with contract storage
var getContractStorage = function () {
    return {};
};


/// Should be called to evaluate single expression
/// Is internally using javascript's 'eval' method
/// Should be checked if it is safe
var evaluateExpression = function (expression) {

    // in future may be replaced with getting storage values based on expression
    var storage = getContractStorage(); 

    copyToGlobalContext(storage);

    // TODO: check if it is safe
    return eval(expression).toString();
};

