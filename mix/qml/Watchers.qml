import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import Qt.labs.settings 1.0
import org.ethereum.qml.QEther 1.0
import "js/Debugger.js" as Debugger
import "js/ErrorLocationFormater.js" as ErrorLocationFormater
import "js/TransactionHelper.js" as TransactionHelper
import "js/QEtherHelper.js" as QEtherHelper
import "."

ColumnLayout {

	property variant tx
	property variant state

	function updateWidthTx(_tx, _state)
	{
		console.log("update tx")
		console.log(JSON.stringify(tx))
		console.log(JSON.stringify(state))
		txLabel.text = tx.label
		tx = _tx
		state = _state
	}

	RowLayout
	{
		Label {
			id: txLabel
		}
	}

	KeyValuePanel
	{
		id: inputParams
		title: qsTr("INPUT PARAMETERS")
	}

	KeyValuePanel
	{
		id: returnParams
		title: qsTr("RETURN PARAMETERS")
	}

	KeyValuePanel
	{
		id: balance
		title: qsTr("BALANCES")
	}

	KeyValuePanel
	{
		id: events
		title: qsTr("EVENTS")
	}
}
