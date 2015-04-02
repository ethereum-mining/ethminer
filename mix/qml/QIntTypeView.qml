import QtQuick 2.0
import QtQuick.Layouts 1.1

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
			anchors.fill: parent
			font.family: boldFont.name
			clip: true
			MouseArea {
				id: mouseArea
				anchors.fill: parent
				hoverEnabled: true
				onClicked: textinput.forceActiveFocus()
			}
		}
	}
}



