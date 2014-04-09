import QtQml 2.2
import QtQuick 2.1
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import Qt.labs.settings 1.0
import org.ethereum 1.0

Item {
	id: main
	anchors.fill: parent
	anchors.margins: 9

//	Qt.application.name: "Walleth"
//	Qt.application.organization: "Ethereum"
//	Qt.application.domain: "org.ethereum"

	Ethereum {
		id: eth
	}

	Account {
		id: myAccount
		address: Key.addressOf("84fc4ba9373c30bfe32d8c5a502854e7f1175878")
		ethereum: eth
		// TODO: state: eth.latest	// could be eth.pending
		// will provide balance, txCount, isContract, incomingTransactions (list model), outgoingTransactions (list model).
		// transaction lists' items will provide value, from, to, final balance.
	}

	// KeyPair provides makeTransaction(recvAddress, value, data (array))

	Text {
		text: "Balance: " + Balance.stringOf(myAccount.balance) + " [" + myAccount.txCount + "]" + "\nAccount: " + Key.stringOf(myAccount.address)
		Layout.minimumHeight: 30
		Layout.fillHeight: true
		Layout.fillWidth: true
	}
}
