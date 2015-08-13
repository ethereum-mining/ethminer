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

Rectangle {
	color: selectedBlockColor
	property variant tx
	property variant currentState
	property variant bc
	property var blockIndex
	property var txIndex

	property string selectedBlockColor: "#accbf2"
	property string selectedBlockForeground: "#445e7f"

	function clear()
	{
		from.text = ""
		to.text = ""
		value.text = ""
		inputParams.clear()
		returnParams.clear()
		accounts.clear()
		events.clear()
	}

	function addAccount(address, amount)
	{
		accounts.add(address, amount)
	}

	function updateWidthTx(_tx, _state, _blockIndex, _txIndex)
	{
		from.text = clientModel.resolveAddress(_tx.sender)
		to.text = _tx.label
		value.text = _tx.value.format()
		tx = _tx
		blockIndex  = _blockIndex
		txIndex = _txIndex
		currentState = _state
		inputParams.init()
		if (_tx.isContractCreation)
		{
			returnParams.role = "creationAddr"
			returnParams._data = {
				creationAddr : {					
				}
			}
			returnParams._data.creationAddr[qsTr("contract address")] = _tx.returned
		}
		else
		{
			returnParams.role = "returnParameters"
			returnParams._data = tx
		}
		returnParams.init()
		accounts.init()
		events.init()
	}

	Column {
		anchors.fill: parent
		spacing: 15
		Rectangle
		{
			height: 15
			width: parent.width - 30
			color: "transparent"
			Row
			{
				id: rowHeader
				anchors.horizontalCenter: parent.horizontalCenter
				anchors.top: rowHeader.parent.top
				anchors.topMargin: 6
				spacing: 5
				Label {
					id: fromLabel
					text: qsTr("from")
					visible: false
					color: selectedBlockForeground					
				}
				Label {
					id: from
					color: selectedBlockForeground
					elide: Text.ElideRight
					maximumLineCount: 1
					clip: true
					width: 200
				}
				Label {
					id: toLabel
					text: qsTr("to")
					visible: false
					color: selectedBlockForeground
				}
				Label {
					id: to
					color: selectedBlockForeground
					elide: Text.ElideRight
					maximumLineCount: 1
					clip: true
					width: 100
				}
				Label {
					id: value
					color: selectedBlockForeground
					font.italic: true
					clip: true
				}
			}

			Image {
				anchors.right: rowHeader.parent.right
				anchors.top: rowHeader.parent.top
				anchors.topMargin: 5
				source: "qrc:/qml/img/edit_combox.png"
				height: 15
				fillMode: Image.PreserveAspectFit
				visible: from.text !== ""
				MouseArea
				{
					anchors.fill: parent
					onClicked:
					{
						bc.blockChainRepeater.editTx(blockIndex, txIndex)
					}
				}
			}
		}

		Rectangle {
			height: 1
			width: parent.width - 30
			anchors.horizontalCenter: parent.horizontalCenter
			border.color: "#cccccc"
			border.width: 1
		}

		KeyValuePanel
		{
			height: 150
			width: parent.width - 30
			anchors.horizontalCenter: parent.horizontalCenter
			id: inputParams
			title: qsTr("INPUT PARAMETERS")
			role: "parameters"
			_data: tx
		}

		KeyValuePanel
		{
			height: 150
			width: parent.width - 30
			anchors.horizontalCenter: parent.horizontalCenter
			id: returnParams
			title: qsTr("RETURN PARAMETERS")
			role: "returnParameters"
			_data: tx
		}

		KeyValuePanel
		{
			height: 150
			width: parent.width - 30
			anchors.horizontalCenter: parent.horizontalCenter
			id: accounts
			title: qsTr("ACCOUNTS")
			role: "accounts"
			_data: currentState
		}

		KeyValuePanel
		{
			height: 150
			width: parent.width - 30
			anchors.horizontalCenter: parent.horizontalCenter
			id: events
			title: qsTr("EVENTS")
			function computeData()
			{
				model.clear()
				var ret = []
				for (var k in tx.logs)
				{
					var param = ""
					for (var p in tx.logs[k].param)
					{
						param += " " + tx.logs[k].param[p].value + " "
					}
					param = "(" + param + ")"
					model.append({ "key": tx.logs[k].name, "value": param })
				}
			}
		}
	}
}
