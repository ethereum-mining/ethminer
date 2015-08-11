import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Dialogs 1.2
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0
import "."

Dialog {
	id: stateListContainer
	modality: Qt.WindowModal
	width: 640
	height: 480
	visible: false
	contentItem: Rectangle {
		anchors.fill: parent
		ColumnLayout
		{
			anchors.fill: parent
			anchors.margins: 10
			TableView {
				id: list
				Layout.fillHeight: true
				Layout.fillWidth: true
				model: projectModel.stateListModel
				itemDelegate: renderDelegate
				headerDelegate: null
				frameVisible: false
				TableViewColumn {
					role: "title"
					title: qsTr("Scenario")
					width: list.width
				}
			}

			Row{
				spacing: 5
				anchors.bottom: parent.bottom
				anchors.right: parent.right
				anchors.rightMargin: 10
				Button {
					action: closeAction
				}
			}
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
					text: qsTr("Edit Genesis");
					Layout.fillHeight: true
					onClicked: list.model.editState(styleData.row);
				}
				ToolButton {
					visible: list.model.defaultStateIndex !== styleData.row
					text: qsTr("Delete");
					Layout.fillHeight: true
					onClicked: list.model.deleteState(styleData.row);
				}
			}
		}
	}

	Row
	{
		Action {
			id: closeAction
			text: qsTr("Close")
			onTriggered: stateListContainer.close();
		}
	}
}

