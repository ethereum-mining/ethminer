function createEther(_value, _unit, _parent)
{
	var etherComponent = Qt.createComponent("qrc:/qml/EtherValue.qml");
	var ether = etherComponent.createObject();
	ether.setValue(_value);
	ether.setUnit(_unit);
	return ether;
}
