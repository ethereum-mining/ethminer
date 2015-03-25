# natspec.js
Javascript Library used to evaluate natspec expressions

[![Build Status][travis-image]][travis-url] [![Coverage Status][coveralls-image]][coveralls-url]

[travis-image]: https://travis-ci.org/ethereum/natspec.js.svg
[travis-url]: https://travis-ci.org/ethereum/natspec.js
[coveralls-image]: https://coveralls.io/repos/ethereum/natspec.js/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/r/ethereum/natspec.js?branch=master

## Usage

It exposes global object `natspec` with method `evaluateExpression`.

```javascript
var natspec = require('natspec');

var natspecExpression = "Will multiply `a` by 7 and return `a * 7`.";
var call = {
    method: 'multiply',
    abi: abi,
    transaction: transaction
};

var evaluatedExpression = natspec.evaluateExpression(natspecExpression, call);
console.log(evaluatedExpression); // "Will multiply 4 by 7 and return 28."
```

More examples are available [here](https://github.com/ethereum/natspec.js/blob/master/test/test.js).

## Building

```bash
npm run-script build
```

## Testing (mocha)

```bash
npm test
```

## Wiki

* [Ethereum Natural Specification Format](https://github.com/ethereum/wiki/wiki/Ethereum-Natural-Specification-Format)
* [Natspec Example](https://github.com/ethereum/wiki/wiki/Natspec-Example)

