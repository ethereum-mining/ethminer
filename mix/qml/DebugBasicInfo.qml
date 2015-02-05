import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1

RowLayout {
	property string titleStr

	function update(_value)
	{
		currentStepValue.text = _value;
	}

	Rectangle {
		width: 120
		height: parent.height
		color: "#e5e5e5"
		Text
		{
			id: title
			font.pixelSize: 12
			anchors.centerIn: parent
			color: "#a2a2a2"
			font.family: "Sans Serif"
			text: titleStr
		}
	}
	Text
	{
		font.pixelSize: 13
		id: currentStepValue
	}
}
