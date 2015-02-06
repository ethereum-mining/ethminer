import QtQuick 2.2
import QtQuick.Controls.Styles 1.1
import QtQuick.Controls 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1

Item {
	Action {
		id: addStateAction
		text: "Add State"
		shortcut: "Ctrl+Alt+T"
		enabled: codeModel.hasContract && !clientModel.running;
		onTriggered: projectModel.stateListModel.addState();
	}
	Action {
		id: editStateAction
		text: "Edit State"
		shortcut: "Ctrl+Alt+T"
		enabled: codeModel.hasContract && !clientModel.running && statesCombo.currentIndex >= 0 && projectModel.stateListModel.count > 0;
		onTriggered: projectModel.stateListModel.editState(statesCombo.currentIndex);
	}

	ColumnLayout {
		anchors.fill: parent
		RowLayout {

			ComboBox {
				id: statesCombo
				model: projectModel.stateListModel
				width: 150
				editable: false
				textRole: "title"
				onActivated:  {
					model.runState(index);
				}
				Connections {
					target: projectModel.stateListModel
					onStateRun: {
						if (statesCombo.currentIndex !== index)
							statesCombo.currentIndex = index;
					}
				}
			}
			Button
			{
				anchors.rightMargin: 9
				anchors.verticalCenter: parent.verticalCenter
				action: editStateAction
			}
			Button
			{
				anchors.rightMargin: 9
				anchors.verticalCenter: parent.verticalCenter
				action: addStateAction
			}
			Button
			{
				anchors.rightMargin: 9
				anchors.verticalCenter: parent.verticalCenter
				action: mineAction
			}

			CheckBox {
				id: recording
				text: qsTr("Record transactions");
				checked: true
				Layout.fillWidth: true


			}
		}
		TableView {
			Layout.fillWidth: true
			Layout.fillHeight: true
			model: logModel

			TableViewColumn {
				role: "block"
				title: qsTr("Block")
				width: 40
			}
			TableViewColumn {
				role: "tindex"
				title: qsTr("Index")
				width: 40
			}
			TableViewColumn {
				role: "contract"
				title: qsTr("Contract")
				width: 100
			}
			TableViewColumn {
				role: "function"
				title: qsTr("Function")
				width: 120
			}
			TableViewColumn {
				role: "value"
				title: qsTr("Value")
				width: 60
			}
			TableViewColumn {
				role: "address"
				title: qsTr("Address")
				width: 120
			}
			TableViewColumn {
				role: "returned"
				title: qsTr("Returned")
				width: 120
			}
			onActivated:  {
				var item = logModel.get(row);
				clientModel.debugTransaction(item.block, item.tindex);
			}
			Keys.onPressed: {
				if ((event.modifiers & Qt.ControlModifier) && event.key === Qt.Key_C && currentRow >=0 && currentRow < logModel.count) {
					var item = logModel.get(currentRow);
					appContext.toClipboard(item.returned);
				}
			}
		}
	}

	ListModel {
		id: logModel
	}

	Connections {
		target: clientModel
		onStateCleared: {
			logModel.clear();
		}
		onNewTransaction: {
			if (recording.checked)
				logModel.append(_tr);
		}
	}

}
