function createEther(_value, _unit, _parent)
{
	var etherComponent = Qt.createComponent("qrc:/qml/EtherValue.qml");
	var ether = etherComponent.createObject();
	ether.setValue(_value);
	ether.setUnit(_unit);
	return ether;
}

function createBigInt(_value)
{
	var bigintComponent = Qt.createComponent("qrc:/qml/BigIntValue.qml");
	var bigint = bigintComponent.createObject();
	bigint.setValue(_value);
	return bigint;
}

function createString(_value)
{
	var stringComponent = Qt.createComponent("qrc:/qml/QStringType.qml");
	var stringC = stringComponent.createObject();
	stringC.setValue(_value);
	return stringC;
}

function createHash(_value)
{
	var hComponent = Qt.createComponent("qrc:/qml/QHashType.qml");
	var hC = hComponent.createObject();
	hC.setValue(_value);
	return hC;
}

