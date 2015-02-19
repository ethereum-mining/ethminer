import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0
import "."

Window {
	id: stateListContainer
	modality: Qt.WindowModal

	width: 640
	height: 480

	visible: false
	ColumnLayout
	{
		anchors.fill: parent
		TableView {
			id: list
			Layout.fillHeight: true
			Layout.fillWidth: true
			model: projectModel.stateListModel
			itemDelegate: renderDelegate
			headerDelegate: null
			TableViewColumn {
				role: "title"
				title: qsTr("State")
				width: list.width
			}
		}

		Button {
			anchors.bottom: parent.bottom
			action: addStateAction
		}
	}

	Component {
		id: renderDelegate
		Item {
			RowLayout {
				anchors.fill: parent
				Text {
					Layout.fillWidth: true
					Layout.fillHeight: true
					text: styleData.value
					font.pointSize: StateStyle.general.basicFontSize
					verticalAlignment: Text.AlignBottom
				}
				ToolButton {
					text: qsTr("Edit");
					Layout.fillHeight: true
					onClicked: list.model.editState(styleData.row);
				}
				ToolButton {
					visible: list.model.defaultStateIndex !== styleData.row
					text: qsTr("Delete");
					Layout.fillHeight: true
					onClicked: list.model.deleteState(styleData.row);
				}
				ToolButton {
					text: qsTr("Run");
					Layout.fillHeight: true
					onClicked: list.model.runState(styleData.row);
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

