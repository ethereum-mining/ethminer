/*
    This file is part of natspec.js.

    natspec.js is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    natspec.js is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with natspec.js.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file natspec.js
 * @authors:
 *   Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 */

var abi = require('./node_modules/ethereum.js/lib/abi.js'); 

/**
 * This object should be used to evaluate natspec expression
 * It has one method evaluateExpression which shoul be used
 */
var natspec = (function () {
    /**
     * Helper method
     * Should be called to copy values from object to global context
     *
     * @method copyToContext
     * @param {Object} object from which we want to copy properties
     * @param {Object} object to which we copy
     */
    var copyToContext = function (obj, context) {
        Object.keys(obj).forEach(function (key) {
            context[key] = obj[key];
        });
    }
    
    /**
     *  Should be used to generate codes, which will be evaluated
     *
     *  @method generateCode
     *  @param {Object} object from which code will be generated
     *  @return {String} javascript code which is used to initalized variables
     */
    var generateCode = function (obj) {
        return Object.keys(obj).reduce(function (acc, key) {
            return acc + "var " + key + " = context['" + key + "'];\n";
        }, "");
    };

    /**
     * Helper method
     * Should be called to get method with given name from the abi
     *
     * @method getMethodWithName
     * @param {Array} contract's abi
     * @param {String} name of the method that we are looking for
     * @return {Object} abi for method with name
     */
    var getMethodWithName = function(abi, name) {
        return abi.filter(function (method) {
            return method.name === name;
        })[0];
    };

    /**
     * Should be used to get all contract method input variables
     *
     * @method getMethodInputParams
     * @param {Object} abi for certain method
     * @param {Object} transaction object
     * @return {Object} object with all contract's method input variables
     */
    var getMethodInputParams = function (method, transaction) {
        // do it with output formatter (cause we have to decode)
        var params = abi.formatOutput(method.inputs, '0x' + transaction.params[0].data.slice(10)); 

        return method.inputs.reduce(function (acc, current, index) {
            acc[current.name] = params[index];
            return acc;
        }, {});
    };
    
    /**
     * Should be called when we want to evaluate natspec expression
     * Replaces all natspec 'subexpressions' with evaluated value
     *
     * @method mapExpressionToEvaluate
     * @param {String} expression to evaluate
     * @param {Function} callback which is called to evaluate te expression
     * @return {String} evaluated expression
     */
    var mapExpressionsToEvaluate = function (expression, cb) {
        var evaluatedExpression = "";

        // match everything in `` quotes
        var pattern = /\`(?:\\.|[^`\\])*\`/gim
        var match;
        var lastIndex = 0;
        try {
            while ((match = pattern.exec(expression)) !== null) {
                var startIndex = pattern.lastIndex - match[0].length;
                var toEval = match[0].slice(1, match[0].length - 1);
                evaluatedExpression += expression.slice(lastIndex, startIndex);
                var evaluatedPart = cb(toEval);
                evaluatedExpression += evaluatedPart;
                lastIndex = pattern.lastIndex;
            }
            
            evaluatedExpression += expression.slice(lastIndex);
        }
        catch (err) {
            throw new Error("Natspec evaluation failed, wrong input params");
        }
    
        return evaluatedExpression;
    };

    /** 
     * Should be called to evaluate single expression
     * Is internally using javascript's 'eval' method
     *
     * @method evaluateExpression
     * @param {String} expression which should be evaluated
     * @param {Object} [call] object containing contract abi, transaction, called method
     * @return {String} evaluated expression
     * @throws exception if method is not found or we are trying to evaluate input params that does not exists
     */
    var evaluateExpression = function (expression, call) {
        //var self = this;
        var context = {};
        
        if (!!call) {
            try {
                var method = getMethodWithName(call.abi, call.method);
                var params = getMethodInputParams(method, call.transaction); 
                copyToContext(params, context);
            }
            catch (err) {
                throw new Error("Natspec evaluation failed, method does not exist");
            }
        }

        var code = generateCode(context);

        var evaluatedExpression = mapExpressionsToEvaluate(expression, function (toEval) {
            var fn = new Function("context", code + "return " + toEval + ";");
            return fn(context).toString();
        });

        return evaluatedExpression;
    };

    /** 
     * Safe version of evaluateExpression
     * Instead of throwing an exception it returns it as a string 
     *
     * @method evaluateExpressionSafe
     * @param {String} expression which should be evaluated
     * @param {Object} [call] object containing contract abi, transaction, called method
     * @return {String} evaluated expression
     */
    var evaluateExpressionSafe = function (expression, call) {
        try {
            return evaluateExpression(expression, call);    
        }
        catch (err) {
            return err.message;
        }
    };

    return {
        evaluateExpression: evaluateExpression,
        evaluateExpressionSafe: evaluateExpressionSafe
    };

})();

module.exports = natspec; 

