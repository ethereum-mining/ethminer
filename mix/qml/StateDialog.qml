import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0
import org.ethereum.qml.QEther 1.0

Window {
	id: modalStateDialog
	modality: Qt.WindowModal

	width:640
	height:480

	visible: false

	property alias stateTitle: titleField.text
	property alias stateBalance: balanceField.value
	property int stateIndex
	property var stateTransactions: []
	signal accepted

	function open(index, item) {
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
		titleField.focus = true;
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

	GridLayout {
		id: dialogContent
		columns: 2
		anchors.fill: parent
		anchors.margins: 10
		rowSpacing: 10
		columnSpacing: 10

		Label {
			text: qsTr("Title")
		}
		TextField {
			id: titleField
			focus: true
			Layout.fillWidth: true
		}

		Label {
			text: qsTr("Balance")
		}
		Ether {
			id: balanceField
			edit: true
			displayFormattedValue: true
			Layout.fillWidth: true
		}

		Label {
			text: qsTr("Transactions")
		}
		ListView {
			Layout.fillWidth: true
			Layout.fillHeight: true
			model: transactionsModel
			delegate: transactionRenderDelegate
		}

		Label {

		}
		Button {
			text: qsTr("Add")
			onClicked: transactionsModel.addTransaction()
		}
	}

	RowLayout {
		anchors.bottom: parent.bottom
		anchors.right: parent.right;

		Button {
			text: qsTr("Ok");
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

	ListModel {
		id: transactionsModel

		function editTransaction(index) {
			transactionDialog.open(index, transactionsModel.get(index));
		}

		function ether(_value, _unit)
		{
			var etherComponent = Qt.createComponent("qrc:/qml/EtherValue.qml");
			var ether = etherComponent.createObject(modalStateDialog);
			ether.setValue(_value);
			ether.setUnit(_unit);
			return ether;
		}

		function addTransaction() {

			// Set next id here to work around Qt bug
			// https://bugreports.qt-project.org/browse/QTBUG-41327
			// Second call to signal handler would just edit the item that was just created, no harm done
			var item = {
				value: ether("0", QEther.Wei),
				functionId: "",
				gas: ether("125000", QEther.Wei),
				gasPrice: ether("100000", QEther.Wei)
			};

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
				}
				ToolButton {
					text: qsTr("Edit");
					Layout.fillHeight: true
					onClicked: transactionsModel.editTransaction(index)
				}
				ToolButton {
					text: qsTr("Delete");
					Layout.fillHeight: true
					onClicked: transactionsModel.deleteTransaction(index)
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
				stateTransactions[index] = item;
			} else {
				transactionsModel.append(item);
				stateTransactions.push(item);
			}
		}
	}

}
