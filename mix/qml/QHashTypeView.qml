import QtQuick 2.0

Item
{
	property alias text: textinput.text
	id: editRoot
	Rectangle {
		anchors.fill: parent
		TextInput {
			id: textinput
			text: text
			anchors.fill: parent
			wrapMode: Text.WrapAnywhere
			MouseArea {
				id: mouseArea
				anchors.fill: parent
				hoverEnabled: true
				onClicked: textinput.forceActiveFocus()
			}
		}
	}
}
