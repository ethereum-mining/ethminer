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

		var setValue = "1"
		if (value === "")
			setValue = parseInt(defaultValue);
		else
			setValue = parseInt(value);
		boolCombo.checked = setValue === "1" ? true: false
		boolCombo.enabled = !readOnly;
	}

	Rectangle {
		color: "transparent"
		anchors.fill: parent
		CheckBox
		{
			property bool inited;
			Component.onCompleted:
			{
				init();
				inited = true;
			}

			id: boolCombo
			anchors.fill: parent
			onCheckedChanged:
			{
				if (inited)
					value = checked ? "1" : "0"

			}
			text: qsTr("True")
		}
	}
}



