function formatLocation(raw, shortMessage)
{
	var splitted = raw.split(':');
	if (!shortMessage)
		return qsTr("Error in line ") + splitted[1] + ", " + qsTr("character ") + splitted[2];
	else
		return "L" + splitted[1] + "," + "C" + splitted[2];
}

function extractErrorInfo(raw, shortMessage)
{
	var _return = {};
	var detail = raw.split('\n')[0];
	var reg = detail.match(/:\d+:\d+:/g);
	if (reg !== null)
	{
		_return.errorLocation = ErrorLocationFormater.formatLocation(reg[0], shortMessage);
		_return.errorDetail = detail.replace(reg[0], "");
		_return.line = reg[0].split(':')[1];
		_return.column = reg[0].split(':')[2];
	}
	else
	{
		_return.errorLocation = "";
		_return.errorDetail = detail;
		_return.line = "";
		_return.column = "";
	}
	return _return;
}
