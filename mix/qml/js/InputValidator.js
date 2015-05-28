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
			if (type.indexOf("int") !== -1)
				res = validateInt(type, values[model[k].name]);
			else if (type.indexOf("bytes") !== -1)
				res = validateBytes(type, values[model[k].name]);
			else if (type.indexOf("bool") !== -1)
				res = validateBool(type, values[model[k].name]);
			else if (type.indexOf("address") !== -1)
				res = validateAddress(type, values[model[k].name]);
			else
				res = validateAddress(type, values[model[k].name]); //we suppose that this is a ctr type.
			if (!res.valid)
				inError.push({ type: type, value: values, message: res.message });
		}
	}
	return inError;
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
			return false;
		}
	}
	ret.valid = nbRegEx.test(_value);
	if (!ret.valid)
		ret.message = _value + " does not represent " + _type + " type.";
	else
	{
		var t = _type.replace("uint", "").replace("int", "");
		var max = parseInt(t) / 4;
		if (_value.length > max)
		{
			ret.valid = false;
			ret.message = _type + " should not contains more than " + max + " digits";
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
		console.log(JSON.stringify(v));
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

