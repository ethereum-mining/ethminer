import QtQuick 2.0

Item
{
	property alias text: textinput.text
	id: editRoot

	SourceSansProBold
	{
		id: boldFont
	}

	Rectangle {
		anchors.fill: parent
		radius: 4
		color: "#f7f7f7"
		TextInput {
			id: textinput
			text: text
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



