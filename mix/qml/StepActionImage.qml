import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1


Rectangle {
	id: buttonActionContainer
	property string disableStateImg
	property string enabledStateImg
	signal clicked

	function enabled(state)
	{
		buttonAction.enabled = state;
		if (state)
			debugImg.iconSource = enabledStateImg;
		else
			debugImg.iconSource = disableStateImg;
	}

	width: debugImg.width + 4
	height: debugImg.height
	color: "transparent"
	Button
	{
		anchors.fill: parent
		id: debugImg
		iconSource: enabledStateImg
		action: buttonAction
		width: 17
		height: 27
	}
	Action {
		id: buttonAction
		onTriggered: {
			buttonActionContainer.clicked();
		}
	}
}
