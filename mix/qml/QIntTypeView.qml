import QtQuick 2.0

Item
{
	property alias value: textinput.text
	property alias readOnly: textinput.readOnly
	id: editRoot
	width: readOnly ? textinput.implicitWidth : 150

	DebuggerPaneStyle {
		id: dbgStyle
	}

	Rectangle {
		anchors.fill: parent
		radius: 4
		TextInput {
			anchors.verticalCenter: parent.verticalCenter
			id: textinput
			font.family: dbgStyle.general.basicFont
			clip: true
			selectByMouse: true
			text: value
			anchors.fill: parent
			font.pointSize: dbgStyle.general.basicFontSize
			color: dbgStyle.general.basicColor
			MouseArea {
				id: mouseArea
				anchors.fill: parent
				hoverEnabled: true
				onClicked: textinput.forceActiveFocus()
			}
		}
	}
}



