Qt.include("QEtherHelper.js")

var nbRegEx;
var arrayRegEx;
var capturenbRegEx;

function validate(model, values)
{
	var inError = [];
	for (var k in model)
	{
		init()
		if (values[model[k].name])
		{
			var type = model[k].type.name;
			var value = values[model[k].name];
			var res = check(type, value)
			if (!res.valid)
				inError.push({ type: type, value: values, message: res.message });
		}
	}
	return inError;
}

function init()
{
	nbRegEx = new RegExp('^[0-9]+$');
	arrayRegEx = new RegExp('\\[[^\\]]*\\]', "g");
	capturenbRegEx = new RegExp("[0-9]+");
}

function check(type, value)
{
	var res = { valid: true, message : "" }

	if (isArray(type))
		res = validateArray(type, value);
	else if (isContractType(type))
		res = validateAddress(type, value);
	else if (type.indexOf("int") !== -1)
		res = validateInt(type, value);
	else if (type.indexOf("enum") !== -1)
		res = validateInt(type, value);
	else if (type.indexOf("bytes") !== -1)
		res = validateBytes(type, value);
	else if (type.indexOf("bool") !== -1)
		res = validateBool(type, value);
	else if (type.indexOf("address") !== -1)
		res = validateAddress(type, value);
	else
	{
		res.valid = true
		res.message = ""
	}
	return res;
}

function isArray(_type)
{
	return arrayRegEx.test(_type);
}

function checkArrayRecursively(_type, _dim, _array)
{
	if (_array instanceof Array)
	{
		if (_dim.length === 0)
			return { valid: false, message: "Your input contains too many dimensions" }
		var size = -1
		var infinite = false
		if (_dim === "[]")
			infinite = true
		else
			size = parseInt(capturenbRegEx.exec(_dim[0]))
		if (_array.length > size && !infinite)
			return { valid: false, message: "Array size does not correspond. Should be " + _dim[0] }
		if (_array.length > 0)
		{
			var _newdim = _dim.slice(0)
			_newdim.splice(0, 1)
			for (var i = 0; i < _array.length; i++)
			{
				var ret = checkArrayRecursively(_type, _newdim, _array[i])
				if (!ret.valid)
					return ret
			}
		}
		return { valid: true, message: "" }
	}
	else
	{
		if (_dim.length > 0)
			return { valid: false, message: "Your input contains too few dimensions" }
		if (typeof(_array) === "number")
			_array = '' + _array + ''
		return check(_type, _array)
	}
}

function validateArray(_type, _value)
{
	try
	{
		_value = JSON.parse(_value)
	}
	catch (e)
	{
		return { valid: false, message: "Input must be JSON formatted like [1,5,3,9] or [[4,9],[4,9],[4,9],[4,9]]" }
	}
	var dim = _type.match(arrayRegEx)
	dim.reverse();
	for (var k = 0; k < dim.length; k++)
		_type = _type.replace(dim[k], "")
	_type = _type.replace(/calldata/g, "")
	return checkArrayRecursively(_type, dim, _value)
}

function isContractType(_type)
{
	for (var k in Object.keys(codeModel.contracts))
	{
		if ("contract " + Object.keys(codeModel.contracts)[k] === _type)
			return true;
	}
	return false;
}

function validateInt(_type, _value)
{
	var ret = { valid: true, message: "" }
	if (_value.indexOf("-") === 0)
	{
		_value = _value.substring(1);
		if (_type.indexOf("uint") === -1)
		{
			ret.valid = false;
			ret.message = "uint type cannot represent negative number";
		}
	}
	ret.valid = nbRegEx.test(_value);
	if (!ret.valid)
		ret.message = _value + " does not represent " + _type + " type.";
	else
	{
		var bigInt = createBigInt(_value);
		bigInt.setBigInt(_value);
		var result = bigInt.checkAgainst(_type);
		if (!result.valid)
		{
			ret.valid = false;
			ret.message = _type + " should be between " + result.minValue + " and " + result.maxValue;
		}
	}
	return ret;
}

function validateAddress(_type, _value)
{
	var ret = { valid: true, message: "" }
	if (_value.indexOf("<") === 0 && _value.indexOf(">") === _value.length - 1)
	{
		var v = _value.split(' - ');
		if (v.length !== 2 || !nbRegEx.test(v[1].replace(">", ""))) // <Contract - 2>
		{
			ret.valid = false;
			ret.message = _value + " is not a valid token for address type.";
		}
	}
	else if (_value.indexOf("0x") !== 0)
	{
		ret.valid = false
		ret.message = "Address type should start with 0x.";
	}
	else
	{
		_value = _value.substring(2);
		if (_value.length !== 40)
		{
			ret.valid = false
			ret.message = "Address type should contain 40 characters.";
		}
	}
	return ret;
}

function validateBytes(_type, _value)
{
	var ret = { valid: true, message: "" }
	if (_type !== "bytes" && _value.length > parseInt(_type.replace("bytes", "")) )
	{
		ret.valid = false;
		ret.message = _type + " should not contains more than " + _type.replace("bytes", "") + " characters";
	}
	return ret;
}

function validateBool(_type, _value)
{
	var ret = { valid: true, message: "" }
	if (!(_value === "1" || _value === "0" || _value === 1 || _value === 0))
	{
		ret.valid = false;
		ret.message = _value + " is not in the correct bool format";
	}
	return ret;
}

function validateEnum(_type, _value)
{
}

