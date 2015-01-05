import QtQuick 2.2
import QtQuick.Controls.Styles 1.2
import QtQuick.Controls 1.2
import QtQuick.Dialogs 1.2
import QtQuick.Layouts 1.1


Rectangle {
	color: "transparent"
	id: transactionListContainer
	focus: true
	anchors.topMargin: 10
	anchors.left: parent.left
	height: parent.height
	width: parent.width

	ListView {
		anchors.top: parent.top
		height: parent.height
		width: parent.width
		id: transactionList
		model: transactionListModel
		delegate: renderDelegate
	}

	Button {
		anchors.bottom: parent.bottom
		text: qsTr("Add")
		onClicked:
		{
			// Set next id here to work around Qt bug
			// https://bugreports.qt-project.org/browse/QTBUG-41327
			// Second call to signal handle would just edit the item that was just created, no harm done
			transactionDialog.reset(transactionListModel.count, transactionListModel);
			transactionDialog.open();
			transactionDialog.focus = true;
		}
	}

	TransactionDialog {
		id: transactionDialog
		onAccepted: {
			transactionListModel.edit(transactionDialog);
		}
	}

	Component {
		id: renderDelegate
		Item {
			id: wrapperItem
			height: 20
			width: parent.width
			RowLayout
			{
				anchors.fill: parent
				Text {
					//anchors.fill: parent
					Layout.fillWidth: true
					Layout.fillHeight: true
					text: title
					font.pointSize: 12
					verticalAlignment: Text.AlignBottom
				}
				ToolButton {
					text: qsTr("Edit");
					Layout.fillHeight: true
					onClicked: {
						transactionDialog.reset(index, transactionListModel);
						transactionDialog.open();
						transactionDialog.focus = true;
					}
				}
				ToolButton {
					text: qsTr("Delete");
					Layout.fillHeight: true
					onClicked: {
					}
				}
				ToolButton {
					text: qsTr("Run");
					Layout.fillHeight: true
					onClicked: {
						transactionListModel.runTransaction(index);
					}
				}
			}
		}
	}
}
