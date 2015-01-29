import QtQuick 2.0
import QtQuick.Controls 1.3

Item
{
	id: editRoot
	property string text
	Rectangle {
		anchors.fill: parent
		ComboBox
		{
			id: boolCombo
			anchors.fill: parent
			onCurrentIndexChanged:
			{
				text = coolComboModel.get(currentIndex).value;
				editRoot.textChanged();
			}
			model: ListModel
			{
			id: coolComboModel
			ListElement { text: qsTr("True"); value: "1" }
			ListElement { text: qsTr("False"); value: "0" }
		}
		}
	}
}



