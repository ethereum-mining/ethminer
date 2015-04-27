import QtQuick 2.0
import QtQuick.Controls 1.3

Item
{
	id: editRoot
	property string value
	property string defaultValue
	property alias readOnly: !boolCombo.enabled
	height: 20
	width: 150

	Rectangle {
		anchors.fill: parent
		ComboBox
		{
			property bool inited: false
			Component.onCompleted:
			{
				if (value === "")
					currentIndex = parseInt(defaultValue);
				else
					currentIndex = parseInt(value);
				inited = true
			}

			id: boolCombo
			anchors.fill: parent
			onCurrentIndexChanged:
			{
				if (inited)
					value = comboModel.get(currentIndex).value;
			}
			model: ListModel
			{
				id: comboModel
				ListElement { text: qsTr("False"); value: "0" }
				ListElement { text: qsTr("True"); value: "1" }
			}
		}
	}
}



