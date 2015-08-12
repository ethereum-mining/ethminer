function solCurrency()
{
	return { "wei": true, "szabo": true, "finney": true, "ether": true };
}

function solKeywords()
{
	return { "break": true, "case": true, "constant": true, "continue": true, "contract": true, "default": true, "delete": true, "do": true, "else": true, "event": true, "external": true, "is": true, "indexed": true, "for": true, "function": true, "if": true, "import": true, "mapping": true, "modifier": true, "new": true, "public": true, "private": true, "internal": true, "return": true, "returns": true, "struct": true, "switch": true, "var": true, "while": true, "enum": true };
}

function solStdContract()
{
	return { "Config": true, "NameReg": true, "CoinReg": true, "owned": true, "onlyowner": true, "named": true, "mortal": true, "coin": true };
}

function solTime()
{
	return { "seconds": true, "minutes": true, "hours": true, "days": true, "weeks": true, "years": true, "after": true };
}

function solTypes()
{
	return { "int": true, "int8": true, "int16": true, "int24": true, "int32": true, "int40": true, "int48": true, "int56": true, "int64": true, "int72": true, "int80": true, "int88": true, "int96": true, "int104": true, "int112": true, "int120": true, "int128": true, "int136": true, "int144": true, "int152": true, "int160": true, "int168": true, "int178": true, "int184": true, "int192": true, "int200": true, "int208": true, "int216": true, "int224": true, "int232": true, "int240": true, "int248": true, "int256": true, "uint": true, "uint8": true, "uint16": true, "uint24": true, "uint32": true, "uint40": true, "uint48": true, "uint56": true, "uint64": true, "uint72": true, "uint80": true, "uint88": true, "uint96": true, "uint104": true, "uint112": true, "uint120": true, "uint128": true, "uint136": true, "uint144": true, "uint152": true, "uint160": true, "uint168": true, "uint178": true, "uint184": true, "uint192": true, "uint200": true, "uint208": true, "uint216": true, "uint224": true, "uint232": true, "uint240": true, "uint248": true, "uint256": true, "bytes0": true, "bytes1": true, "bytes2": true, "bytes3": true, "bytes4": true, "bytes5": true, "bytes6": true, "bytes7": true, "bytes8": true, "bytes9": true, "bytes10": true, "bytes11": true, "bytes12": true, "bytes13": true, "bytes14": true, "bytes15": true, "bytes16": true, "bytes17": true, "bytes18": true, "bytes19": true, "bytes20": true, "bytes21": true, "bytes22": true, "bytes23": true, "bytes24": true, "bytes25": true, "bytes26": true, "bytes27": true, "bytes28": true, "bytes29": true, "bytes30": true, "bytes31": true, "bytes32": true, "bytes": true, "byte": true, "address": true, "bool": true, "string": true, "real": true, "ureal": true };
}

function solMisc()
{
	return { "true": true, "false": true, "null": true };
}

function solBuiltIn()
{
	return { "msg": true, "tx": true, "block": true, "now": true };
}

function solBlock()
{
	return { "coinbase": true, "difficulty": true, "gaslimit": true, "number": true, "blockhash": true, "timestamp":true };
}

function solMsg()
{
	return { "data": true, "gas": true, "sender": true, "sig": true, "value": true };
}

function solTx()
{
	return { "gasprice": true, "origin": true }
}

function keywordsName()
{
	var keywords = {};
	keywords[solCurrency.name.toLowerCase()] = "Currency";
	keywords[solKeywords.name.toLowerCase()] = "Keyword";
	keywords[solStdContract.name.toLowerCase()] = "Contract";
	keywords[solTime.name.toLowerCase()] = "Time";
	keywords[solTypes.name.toLowerCase()] = "Type";
	keywords[solMisc.name.toLowerCase()] = "Misc";
	keywords[solBuiltIn.name.toLowerCase()] = "BuiltIn";
	keywords[solBlock.name.toLowerCase()] = "Block";
	keywords[solMsg.name.toLowerCase()] = "Message";
	keywords[solTx.name.toLowerCase()] = "Transaction";
	return keywords;
}
