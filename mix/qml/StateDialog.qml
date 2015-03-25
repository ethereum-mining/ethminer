import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0
import QtQuick.Controls.Styles 1.3
import org.ethereum.qml.QEther 1.0
import "js/QEtherHelper.js" as QEtherHelper
import "js/TransactionHelper.js" as TransactionHelper
import "."

Window {
	id: modalStateDialog
	modality: Qt.ApplicationModal

	width: 590
	height: 480
	title: qsTr("Edit State")
	visible: false
	color: StateDialogStyle.generic.backgroundColor

	property alias stateTitle: titleField.text
	property alias isDefault: defaultCheckBox.checked
	property int stateIndex
	property var stateTransactions: []
	property var stateAccounts: []
	signal accepted

	function open(index, item, setDefault) {
		stateIndex = index;
		stateTitle = item.title;
		transactionsModel.clear();

		stateTransactions = [];
		var transactions = item.transactions;
		for (var t = 0; t < transactions.length; t++) {
			transactionsModel.append(item.transactions[t]);
			stateTransactions.push(item.transactions[t]);
		}

		accountsModel.clear();
		stateAccounts = [];
		for (var k = 0; k < item.accounts.length; k++)
		{
			accountsModel.append(item.accounts[k]);
			stateAccounts.push(item.accounts[k]);
		}

		modalStateDialog.setX((Screen.width - width) / 2);
		modalStateDialog.setY((Screen.height - height) / 2);

		visible = true;
		isDefault = setDefault;
		titleField.focus = true;
		defaultCheckBox.enabled = !isDefault;
		forceActiveFocus();
	}

	function close() {
		visible = false;
	}

	function getItem() {
		var item = {
			title: stateDialog.stateTitle,
			transactions: [],
			accounts: []
		}
		item.transactions = stateTransactions;
		item.accounts = stateAccounts;
		return item;
	}

	ColumnLayout {
		anchors.fill: parent
		anchors.margins: 10
		ColumnLayout {
			id: dialogContent
			anchors.top: parent.top

			RowLayout
			{
				Layout.fillWidth: true
				DefaultLabel {
					Layout.preferredWidth: 75
					text: qsTr("Title")
				}
				DefaultTextField
				{
					id: titleField
					Layout.fillWidth: true
				}
			}

			CommonSeparator
			{
				Layout.fillWidth: true
			}

			RowLayout
			{
				Layout.fillWidth: true

				Rectangle
				{
					Layout.preferredWidth: 75
					DefaultLabel {
						id: accountsLabel
						Layout.preferredWidth: 75
						text: qsTr("Accounts")
					}

					Button
					{
						anchors.top: accountsLabel.bottom
						anchors.topMargin: 10
						iconSource: "qrc:/qml/img/plus.png"
						action: newAccountAction
					}

					Action {
						id: newAccountAction
						tooltip: qsTr("Add new Account")
						onTriggered:
						{
							var account = stateListModel.newAccount("1000000", QEther.Ether);
							stateAccounts.push(account);
							accountsModel.append(account);
						}
					}
				}

				MessageDialog
				{
					id: alertAlreadyUsed
					text: qsTr("This account is in use. You cannot remove it. The first account is used to deploy config contract and cannot be removed.")
					icon: StandardIcon.Warning
					standardButtons: StandardButton.Ok
				}

				TableView
				{
					id: accountsView
					Layout.fillWidth: true
					model: accountsModel
					headerVisible: false
					TableViewColumn {
						role: "name"
						title: qsTr("Name")
						width: 150
						delegate: Item {
							RowLayout
							{
								height: 25
								width: parent.width
								Button
								{
									iconSource: "qrc:/qml/img/delete_sign.png"
									action: deleteAccountAction
								}

								Action {
									id: deleteAccountAction
									tooltip: qsTr("Delete Account")
									onTriggered:
									{
										if (transactionsModel.isUsed(stateAccounts[styleData.row].secret))
											alertAlreadyUsed.open();
										else
										{
											stateAccounts.splice(styleData.row, 1);
											accountsModel.remove(styleData.row);
										}
									}
								}

								DefaultTextField {
									anchors.verticalCenter: parent.verticalCenter
									onTextChanged: {
										if (styleData.row > -1)
											stateAccounts[styleData.row].name = text;
									}
									text:  {
										return styleData.value
									}
								}
							}
						}
					}

					TableViewColumn {
						role: "balance"
						title: qsTr("Balance")
						width: 200
						delegate: Item {
							Ether {
								id: balanceField
								edit: true
								displayFormattedValue: false
								value: styleData.value
							}
						}
					}
					rowDelegate:
						Rectangle {
						color: styleData.alternate ? "transparent" : "#f0f0f0"
						height: 30;
					}
				}
			}

			CommonSeparator
			{
				Layout.fillWidth: true
			}

			RowLayout
			{
				Layout.fillWidth: true
				DefaultLabel {
					Layout.preferredWidth: 75
					text: qsTr("Default")
				}
				CheckBox {
					id: defaultCheckBox
					Layout.fillWidth: true
				}
			}

			CommonSeparator
			{
				Layout.fillWidth: true
			}
		}

		ColumnLayout {
			anchors.top: dialogContent.bottom
			anchors.topMargin: 5
			spacing: 0
			RowLayout
			{
				Layout.preferredWidth: 150
				DefaultLabel {
					text: qsTr("Transactions: ")
				}

				Button
				{
					iconSource: "qrc:/qml/img/plus.png"
					action: newTrAction
					width: 10
					height: 10
					anchors.right: parent.right
				}

				Action {
					id: newTrAction
					tooltip: qsTr("Create a new transaction")
					onTriggered: transactionsModel.addTransaction()
				}
			}

			ScrollView
			{
				Layout.fillHeight: true
				Layout.preferredWidth: 300
				Column
				{
					Layout.fillHeight: true
					Repeater
					{
						id: trRepeater
						model: transactionsModel
						delegate: transactionRenderDelegate
						visible: transactionsModel.count > 0
						height: 20 * transactionsModel.count
					}
				}
			}

			CommonSeparator
			{
				Layout.fillWidth: true
			}
		}

		RowLayout
		{
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
		id: accountsModel

		function removeAccount(_i)
		{
			accountsModel.remove(_i);
			stateAccounts.splice(_i, 1);
		}
	}

	ListModel {
		id: transactionsModel

		function editTransaction(index) {
			transactionDialog.stateAccounts = stateAccounts;
			transactionDialog.open(index, transactionsModel.get(index));
		}

		function addTransaction() {

			// Set next id here to work around Qt bug
			// https://bugreports.qt-project.org/browse/QTBUG-41327
			// Second call to signal handler would just edit the item that was just created, no harm done
			var item = TransactionHelper.defaultTransaction();
			transactionDialog.stateAccounts = stateAccounts;
			transactionDialog.open(transactionsModel.count, item);
		}

		function deleteTransaction(index) {
			stateTransactions.splice(index, 1);
			transactionsModel.remove(index);
		}

		function isUsed(secret)
		{
			for (var i in stateTransactions)
			{
				if (stateTransactions[i].sender === secret)
					return true;
			}
			return false;
		}
	}

	Component {
		id: transactionRenderDelegate
		RowLayout {
			DefaultLabel {
				Layout.preferredWidth: 150
				text: functionId
			}

			Button
			{
				id: deleteBtn
				iconSource: "qrc:/qml/img/delete_sign.png"
				action: deleteAction
				width: 10
				height: 10
				Action {
					id: deleteAction
					tooltip: qsTr("Delete")
					onTriggered: transactionsModel.deleteTransaction(index)
				}
			}

			Button
			{
				iconSource: "qrc:/qml/img/edit.png"
				action: editAction
				visible: stdContract === false
				width: 10
				height: 10
				Action {
					id: editAction
					tooltip: qsTr("Edit")
					onTriggered: transactionsModel.editTransaction(index)
				}
			}
		}
	}

	TransactionDialog
	{
		id: transactionDialog
		onAccepted:
		{
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
