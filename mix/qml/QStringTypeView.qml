import QtQuick 2.0

Item
{
	property alias value: textinput.text
	property alias readOnly: textinput.readOnly
	id: editRoot
	height: 20
	width: readOnly ? textinput.implicitWidth : 150

	SourceSansProBold
	{
		id: boldFont
	}

	Rectangle {
		anchors.fill: parent
		radius: 4
		TextInput {
			id: textinput
			text: value
			clip: true
			anchors.fill: parent
			wrapMode: Text.WrapAnywhere
			font.family: boldFont.name
			selectByMouse: true
		}
	}
}



