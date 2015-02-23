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

