import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1


Rectangle {
	id: buttonActionContainer
	property string disableStateImg
	property string enabledStateImg
	signal clicked
	width: 15
	height: 15
	color: "transparent"
	Button
	{
		anchors.fill: parent
		id: debugImg
		iconSource: enabledStateImg
		action: buttonAction
		style: ButtonStyle {
			background: Component {
				Rectangle {
					color: "transparent"
					border.width: 0
					width: 15
					height: 15
				}
			}
		}
	}
	Action {
		id: buttonAction
		onTriggered: {
			buttonActionContainer.clicked();
		}
		enabled: codeModel.hasContract && !debugModel.running;
		onEnabledChanged: {
			if (enabled)
				iconSource = enabledStateImg
			else
				iconSource = disableStateImg
		}
	}
}
