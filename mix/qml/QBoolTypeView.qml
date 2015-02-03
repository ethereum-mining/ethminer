import QtQuick 2.0
import QtQuick.Controls 1.3

Item
{
	id: editRoot
	property string text
	property string defaultValue

	Rectangle {
		anchors.fill: parent
		ComboBox
		{
			property bool inited: false
			Component.onCompleted:
			{
				if (text === "")
					currentIndex = parseInt(defaultValue);
				else
					currentIndex = parseInt(text);
				inited = true
			}

			id: boolCombo
			anchors.fill: parent
			onCurrentIndexChanged:
			{
				if (inited)
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



