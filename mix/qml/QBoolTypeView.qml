import QtQuick 2.0
import QtQuick.Controls 1.3

Item
{
	id: editRoot
	property string text
	property bool defaultValue
	Rectangle {
		anchors.fill: parent
		ComboBox
		{
			Component.onCompleted:
			{
				text = (defaultValue ? "1" : "0");
				currentIndex = parseInt(text);
			}

			id: boolCombo
			anchors.fill: parent
			onCurrentIndexChanged:
			{
				text = comboModel.get(currentIndex).value;
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



