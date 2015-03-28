import QtQuick 2.0

Item
{
	property alias value: textinput.text
	id: editRoot
	height: 20
	width: 150

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
			MouseArea {
				id: mouseArea
				anchors.fill: parent
				hoverEnabled: true
				onClicked: textinput.forceActiveFocus()
			}
		}
	}
}



