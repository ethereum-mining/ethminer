import QtQuick 2.0
import QtQuick.Controls 1.3

Item
{
	id: editRoot
	property string value
	property string defaultValue
	property bool readOnly: !boolCombo.enabled
	height: 20
	width: 150

	onReadOnlyChanged: {
		boolCombo.enabled = !readOnly;
	}

	function init()
	{
		value = value === true ? "1" : value
		value = value === false ? "0" : value;
		value = value === "true" ? "1" : value
		value = value === "false" ? "0" : value;

		if (value === "")
			boolCombo.currentIndex = parseInt(defaultValue);
		else
			boolCombo.currentIndex = parseInt(value);
		boolCombo.enabled = !readOnly;
	}

	Rectangle {
		anchors.fill: parent
		ComboBox
		{
			property bool inited;
			Component.onCompleted:
			{
				init();
				inited = true;
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



