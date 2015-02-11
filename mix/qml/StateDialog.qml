import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0
import QtQuick.Controls.Styles 1.3
import org.ethereum.qml.QEther 1.0
import "js/QEtherHelper.js" as QEtherHelper
import "js/TransactionHelper.js" as TransactionHelper
import "."

Window {
	id: modalStateDialog
	modality: Qt.WindowModal

	width: 640
	height: 480

	visible: false
	color: StateDialogStyle.generic.backgroundColor

	property alias stateTitle: titleField.text
	property alias stateBalance: balanceField.value
	property alias isDefault: defaultCheckBox.checked
	property int stateIndex
	property var stateTransactions: []
	signal accepted

	function open(index, item, setDefault) {
		stateIndex = index;
		stateTitle = item.title;
		balanceField.value = item.balance;
		transactionsModel.clear();
		stateTransactions = [];
		var transactions = item.transactions;
		for (var t = 0; t < transactions.length; t++) {
			transactionsModel.append(item.transactions[t]);
			stateTransactions.push(item.transactions[t]);
		}
		visible = true;
		isDefault = setDefault;
		titleField.focus = true;
		defaultCheckBox.enabled = !isDefault;
	}

	function close() {
		visible = false;
	}

	function getItem() {
		var item = {
			title: stateDialog.stateTitle,
			balance: stateDialog.stateBalance,
			transactions: []
		}
		item.transactions = stateTransactions;
		return item;
	}

	SourceSansProRegular
	{
		id: regularFont
	}

	Rectangle
	{
		anchors.fill: parent
		anchors.margins: 10
		color: StateDialogStyle.generic.backgroundColor
		GridLayout {
			id: dialogContent
			columns: 2
			anchors.top: parent.top
			rowSpacing: 10
			columnSpacing: 10

			Label {
				text: qsTr("Title")
				font.family: regularFont.name
				color: "#808080"
			}
			TextField {
				id: titleField
				focus: true
				Layout.fillWidth: true
				font.family: regularFont.name
			}

			Label {
				text: qsTr("Balance")
				font.family: regularFont.name
				color: "#808080"
			}
			Ether {
				id: balanceField
				edit: true
				displayFormattedValue: true
				Layout.fillWidth: true
			}

			Label {
				text: qsTr("Default")
				font.family: regularFont.name
				color: "#808080"
			}
			CheckBox {
				id: defaultCheckBox
				Layout.fillWidth: true
			}
		}

		ColumnLayout
		{
			width: parent.width
			anchors.top: dialogContent.bottom
			anchors.topMargin: 5
			spacing: 5
			Label {
				text: qsTr("Transactions")
				font.family: regularFont.name
				color: "#808080"
			}

			ListView {
				id: trList
				Layout.preferredWidth: 200
				Layout.fillHeight: true
				Layout.minimumHeight: 20 * transactionsModel.count
				model: transactionsModel
				delegate: transactionRenderDelegate
				visible: transactionsModel.count > 0
			}

			ToolButton {
				onClicked: transactionsModel.addTransaction()
				style: ButtonStyle
				{
				 label: Text {
					 font.family: regularFont.name
					 text: qsTr("+")
					 font.pointSize: 20
					 color: "#808080"
				 }
				 background: Rectangle {
					 color: "transparent"
				 }
				}
			}
		}

	RowLayout {
		anchors.bottom: parent.bottom
		anchors.right: parent.right;

		Button {
			text: qsTr("OK");
			onClicked: {
				close();
				accepted();
			}
		}
		Button {
			text: qsTr("Cancel");
			onClicked: close();
		}
	}
}




	ListModel {
		id: transactionsModel

		function editTransaction(index) {
			transactionDialog.open(index, transactionsModel.get(index));
		}

		function addTransaction() {

			// Set next id here to work around Qt bug
			// https://bugreports.qt-project.org/browse/QTBUG-41327
			// Second call to signal handler would just edit the item that was just created, no harm done
			var item = TransactionHelper.defaultTransaction();
			transactionDialog.open(transactionsModel.count, item);
		}

		function deleteTransaction(index) {
			stateTransactions.splice(index, 1);
			transactionsModel.remove(index);
		}
	}

	Component {
		id: transactionRenderDelegate
		Item {
			id: wrapperItem
			height: 20
			width: parent.width
			RowLayout {
				anchors.fill: parent
				Text {
					Layout.fillWidth: true
					Layout.fillHeight: true
					text: functionId
					font.pointSize: 12
					verticalAlignment: Text.AlignBottom
					font.family: regularFont.name
				}
				ToolButton {
					text: qsTr("Edit");
					visible: !stdContract
					Layout.fillHeight: true
					onClicked: transactionsModel.editTransaction(index)
					style: ButtonStyle
					{
					 label: Text {
						 font.family: regularFont.name
						 text: qsTr("Edit")
						 font.italic: true
						 font.pointSize: 9
					 }
					 background: Rectangle {
						 color: "transparent"
					 }
					}
				}
				ToolButton {
					visible: index >= 0 ? !transactionsModel.get(index).executeConstructor : false
					text: qsTr("Delete");
					Layout.fillHeight: true
					onClicked: transactionsModel.deleteTransaction(index)
					style: ButtonStyle
					{
					 label: Text {
						 font.family: regularFont.name
						 text: qsTr("Delete")
						 font.italic: true
						 font.pointSize: 9
					 }
					 background: Rectangle {
						 color: "transparent"
					 }
					}
				}
			}
		}
	}

	TransactionDialog {
		id: transactionDialog
		onAccepted: {
			var item = transactionDialog.getItem();

			if (transactionDialog.transactionIndex < transactionsModel.count) {
				transactionsModel.set(transactionDialog.transactionIndex, item);
				stateTransactions[transactionDialog.transactionIndex] = item;
			} else {
				transactionsModel.append(item);
				stateTransactions.push(item);
			}
		}
	}

}
