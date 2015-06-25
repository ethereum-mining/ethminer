import QtQuick 2.0

Item
{
	property alias value: textinput.text
	property alias readOnly: textinput.readOnly
	id: editRoot
	height: 20
	width: 150

	SourceSansProBold
	{
		id: boldFont
	}

	TextInput {
		id: textinput
		text: value
		wrapMode: Text.WrapAnywhere
		MouseArea {
			id: mouseArea
			anchors.fill: parent
			hoverEnabled: true
			onClicked: textinput.forceActiveFocus()
		}
	}
}
