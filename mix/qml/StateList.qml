import QtQuick 2.2
import QtQuick.Controls.Styles 1.1
import QtQuick.Controls 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1

Rectangle {
	color: "#ededed"
	id: stateListContainer
	focus: true
	anchors.topMargin: 10
	anchors.left: parent.left
	height: parent.height
	width: parent.width

	ListView {
		id: list
		anchors.top: parent.top
		height: parent.height
		width: parent.width
		model: projectModel.stateListModel
		delegate: renderDelegate
	}

	Button {
		anchors.bottom: parent.bottom
		action: addStateAction
	}

	Component {
		id: renderDelegate
		Item {
			id: wrapperItem
			height: 20
			width: parent.width
			RowLayout {
				anchors.fill: parent
				Text {
					Layout.fillWidth: true
					Layout.fillHeight: true
					text: title
					font.pointSize: 12
					verticalAlignment: Text.AlignBottom
				}
				ToolButton {
					text: qsTr("Edit");
					Layout.fillHeight: true
					onClicked: list.model.editState(index);
				}
				ToolButton {
					visible: list.model.count - 1 != index
					text: qsTr("Delete");
					Layout.fillHeight: true
					onClicked: list.model.deleteState(index);
				}
				ToolButton {
					text: qsTr("Run");
					Layout.fillHeight: true
					onClicked: list.model.runState(index);
				}
			}
		}
	}

	Action {
		id: addStateAction
		text: "&Add State"
		shortcut: "Ctrl+T"
		enabled: codeModel.hasContract && !clientModel.running;
		onTriggered: list.model.addState();
	}
}

