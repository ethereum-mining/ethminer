import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1


Rectangle {
	id: buttonActionContainer
	color: "transparent"
	property string disableStateImg
	property string enabledStateImg
	property string buttonTooltip
	property string buttonShortcut
	signal clicked

	function enabled(state)
	{
		buttonAction.enabled = state;
		if (state)
			debugImage.source = enabledStateImg;
		else
			debugImage.source = disableStateImg;
	}

	Button
	{
		anchors.fill: parent
		id: debugImg
/*		iconSource: enabledStateImg
*/		action: buttonAction
	}

	Image {
		id: debugImage
		source: enabledStateImg
		anchors.centerIn: parent
		fillMode: Image.PreserveAspectFit
		width: 15
		height: 15
	}

	Action {
		tooltip: buttonTooltip
		id: buttonAction
		shortcut: buttonShortcut
		onTriggered: {
			buttonActionContainer.clicked();
		}
	}
}
