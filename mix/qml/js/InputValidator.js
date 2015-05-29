Qt.include("QEtherHelper.js")

var nbRegEx = new RegExp('^[0-9]+$');
function validate(model, values)
{
	var inError = [];
	for (var k in model)
	{
		if (values[model[k].name])
		{
			var type = model[k].type.name;
			var res;
			if (isContractType(type))
				res = validateAddress(type, values[model[k].name]);
			else if (type.indexOf("int") !== -1)
				res = validateInt(type, values[model[k].name]);
			else if (type.indexOf("bytes") !== -1)
				res = validateBytes(type, values[model[k].name]);
			else if (type.indexOf("bool") !== -1)
				res = validateBool(type, values[model[k].name]);
			else if (type.indexOf("address") !== -1)
				res = validateAddress(type, values[model[k].name]);
			else
				res.valid = true;
			if (!res.valid)
				inError.push({ type: type, value: values, message: res.message });
		}
	}
	return inError;
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
	if (_value.length > parseInt(_type.replace("bytes", "")) )
	{
		ret.valid = false;
		ret.message = _type + " should not contains more than " + _type.replace("bytes", "") + " characters";
	}
	return ret;
}

function validateBool(_type, _value)
{
	var ret = { valid: true, message: "" }
	if (_value !== "1" && _value !== "0")
	{
		ret.valid = false;
		ret.message = _value + " is not in the correct bool format";
	}
	return ret;
}

function validateEnum(_type, _value)
{
}

