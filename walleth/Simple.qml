import QtQuick 2.1
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
//import org.ethereum 1.0

Label {
	id: main
	anchors.fill: parent
	anchors.margins: 9
	Label {
		text: "Balance: " + u256.stringOf(eth.balance) + "\nAccount: " + key.stringOf(eth.address)
		Layout.minimumHeight: 30
		Layout.fillHeight: true
		Layout.fillWidth: true
	}
}
