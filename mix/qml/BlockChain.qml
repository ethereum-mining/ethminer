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
	id: blockChainPanel
	property variant model
	spacing: 0
	property int previousWidth
	property variant debugTrRequested: []
	signal chainChanged
	signal chainReloaded

	Connections
	{
		target: codeModel
		onContractRenamed: {
			rebuild.startBlinking()
		}
		onNewContractCompiled: {
			rebuild.startBlinking()
		}
	}

	onChainChanged: {
		rebuild.startBlinking()
	}

	onWidthChanged:
	{
		var minWidth = scenarioMinWidth - 20 // margin
		if (width <= minWidth || previousWidth <= minWidth)
		{
			fromWidth = 100
			toWidth = 100
			valueWidth = 200
		}
		else
		{
			var diff = (width - previousWidth) / 3;
			fromWidth = fromWidth + diff < 100 ? 100 : fromWidth + diff
			toWidth = toWidth + diff < 100 ? 100 : toWidth + diff
			valueWidth = valueWidth + diff < 200 ? 200 : valueWidth + diff
		}
		previousWidth = width
	}

	function load(scenario)
	{
		if (!scenario)
			return;
		if (model)
			rebuild.startBlinking()
		model = scenario
		blockModel.clear()
		for (var b in model.blocks)
			blockModel.append(model.blocks[b])
		previousWidth = width
	}

	property int statusWidth: 30
	property int fromWidth: 150
	property int toWidth: 100
	property int valueWidth: 200
	property int logsWidth: 40
	property int debugActionWidth: 40
	property int horizontalMargin: 10
	property int cellSpacing: 10

	RowLayout
	{
		id: header
		spacing: 0
		Layout.preferredHeight: 30
		Rectangle
		{
			Layout.preferredWidth: statusWidth
			Layout.preferredHeight: parent.height
			color: "transparent"
			Image {
				id: debugImage
				anchors.verticalCenter: parent.verticalCenter
				anchors.horizontalCenter: parent.horizontalCenter
				source: "qrc:/qml/img/recycleicon@2x.png"
				width: statusWidth + 20
				fillMode: Image.PreserveAspectFit
			}
		}
		Rectangle
		{
			Layout.preferredWidth: fromWidth
			Label
			{
				anchors.verticalCenter: parent.verticalCenter
				text: "From"
				anchors.left: parent.left
				anchors.leftMargin: horizontalMargin
			}
		}
		Label
		{
			text: "To"
			Layout.preferredWidth: toWidth + cellSpacing
		}
		Label
		{
			text: "Value"
			Layout.preferredWidth: valueWidth + cellSpacing
		}
		Label
		{
			text: "Logs"
			Layout.preferredWidth: logsWidth + cellSpacing
		}
		Label
		{
			text: ""
			Layout.preferredWidth: debugActionWidth
		}
	}

	Rectangle
	{
		Layout.preferredHeight: 500
		Layout.preferredWidth: parent.width
		border.color: "#cccccc"
		border.width: 2
		color: "white"
		ScrollView
		{
			id: blockChainScrollView
			anchors.fill: parent
			anchors.topMargin: 8
			ColumnLayout
			{
				id: blockChainLayout
				width: parent.width
				spacing: 20

				Block
				{
					scenario: blockChainPanel.model
					Layout.preferredWidth: blockChainScrollView.width
					Layout.preferredHeight: 60
					blockIndex: -1
					transactions: []
					status: ""
					number: -2
					trHeight: 60
				}

				Repeater // List of blocks
				{
					id: blockChainRepeater
					model: blockModel
					Block
					{
						scenario: blockChainPanel.model
						Layout.preferredWidth: blockChainScrollView.width
						Layout.preferredHeight:
						{
							return calculateHeight()
						}
						blockIndex: index
						transactions:
						{
							if (index >= 0)
								return blockModel.get(index).transactions
							else
								return []
						}

						status:
						{
							if (index >= 0)
								return blockModel.get(index).status
							else
								return ""
						}

						number:
						{
							if (index >= 0)
								return blockModel.get(index).number
							else
								return 0
						}
					}
				}
			}
		}
	}

	ListModel
	{
		id: blockModel

		function appendBlock(block)
		{
			blockModel.append(block);
		}

		function appendTransaction(tr)
		{
			blockModel.get(blockModel.count - 1).transactions.append(tr)
		}

		function removeTransaction(blockIndex, trIndex)
		{
			blockModel.get(blockIndex).transactions.remove(trIndex)
		}

		function removeLastBlock()
		{
			blockModel.remove(blockModel.count - 1)
		}

		function removeBlock(index)
		{
			blockModel.remove(index)
		}

		function getTransaction(block, tr)
		{
			return blockModel.get(block).transactions.get(tr)
		}

		function setTransaction(blockIndex, trIndex, tr)
		{
			blockModel.get(blockIndex).transactions.set(trIndex, tr)
		}

		function setTransactionProperty(blockIndex, trIndex, propertyName, value)
		{
			blockModel.get(blockIndex).transactions.set(trIndex, { propertyName: value })
		}
	}

	Rectangle
	{
		Layout.preferredWidth: parent.width
		RowLayout
		{
			anchors.horizontalCenter: parent.horizontalCenter
			anchors.top: parent.top
			anchors.topMargin: 10
			spacing: 20

			Rectangle {
				Layout.preferredWidth: 100
				Layout.preferredHeight: 30

				ScenarioButton {
					id: rebuild
					text: qsTr("Rebuild")
					width: 100
					height: 30
					roundLeft: true
					roundRight: true
					onClicked:
					{
						if (ensureNotFuturetime.running)
							return;
						stopBlinking()
						var retBlocks = [];
						var bAdded = 0;
						for (var j = 0; j < model.blocks.length; j++)
						{
							var b = model.blocks[j];
							var block = {
								hash: b.hash,
								number: b.number,
								transactions: [],
								status: b.status
							}
							for (var k = 0; k < model.blocks[j].transactions.length; k++)
							{
								if (blockModel.get(j).transactions.get(k).saveStatus)
								{
									var tr = model.blocks[j].transactions[k]
									tr.saveStatus = true
									block.transactions.push(tr);
								}

							}
							if (block.transactions.length > 0)
							{
								bAdded++
								block.number = bAdded
								block.status = "mined"
								retBlocks.push(block)
							}

						}
						if (retBlocks.length === 0)
							retBlocks.push(projectModel.stateListModel.createEmptyBlock())
						else
						{
							var last = retBlocks[retBlocks.length - 1]
							last.number = -1
							last.status = "pending"
						}

						model.blocks = retBlocks
						blockModel.clear()
						for (var j = 0; j < model.blocks.length; j++)
							blockModel.append(model.blocks[j])

						ensureNotFuturetime.start()
						clientModel.setupScenario(model);
					}
					buttonShortcut: ""
					sourceImg: "qrc:/qml/img/recycleicon@2x.png"
				}
			}


			Rectangle
			{
				Layout.preferredWidth: 200
				Layout.preferredHeight: 30
				color: "transparent"

				ScenarioButton {
					id: addTransaction
					text: qsTr("Add Tx")
					onClicked:
					{
						var lastBlock = model.blocks[model.blocks.length - 1];
						if (lastBlock.status === "mined")
						{
							var newblock = projectModel.stateListModel.createEmptyBlock()
							blockModel.appendBlock(newblock)
							model.blocks.push(newblock);
						}

						var item = TransactionHelper.defaultTransaction()
						transactionDialog.stateAccounts = model.accounts
						transactionDialog.execute = true
						transactionDialog.open(model.blocks[model.blocks.length - 1].transactions.length, model.blocks.length - 1, item)
					}
					width: 100
					height: 30
					buttonShortcut: ""
					sourceImg: "qrc:/qml/img/sendtransactionicon@2x.png"
					roundLeft: true
					roundRight: false
				}

				Timer
				{
					id: ensureNotFuturetime
					interval: 1000
					repeat: false
					running: false
				}

				Rectangle
				{
					width: 1
					height: parent.height
					anchors.right: addBlockBtn.left
					color: "#ededed"
				}

				ScenarioButton {
					id: addBlockBtn
					text: qsTr("Add Block..")
					anchors.left: addTransaction.right
					roundLeft: false
					roundRight: true
					onClicked:
					{
						if (ensureNotFuturetime.running)
							return
						if (clientModel.mining || clientModel.running)
							return
						if (model.blocks.length > 0)
						{
							var lastBlock = model.blocks[model.blocks.length - 1]
							if (lastBlock.status === "pending")
							{
								ensureNotFuturetime.start()
								clientModel.mine()
							}
							else
								addNewBlock()
						}
						else
							addNewBlock()

					}

					function addNewBlock()
					{
						var block = projectModel.stateListModel.createEmptyBlock()
						model.blocks.push(block)
						blockModel.appendBlock(block)
					}
					width: 100
					height: 30

					buttonShortcut: ""
					sourceImg: "qrc:/qml/img/addblock@2x.png"
				}
			}


			Connections
			{
				target: clientModel
				onNewBlock:
				{
					if (!clientModel.running)
					{
						var lastBlock = model.blocks[model.blocks.length - 1]
						lastBlock.status = "mined"
						lastBlock.number = model.blocks.length
						var lastB = blockModel.get(model.blocks.length - 1)
						lastB.status = "mined"
						lastB.number = model.blocks.length
						addBlockBtn.addNewBlock()
					}
				}
				onStateCleared:
				{
				}
				onNewRecord:
				{
					var blockIndex =  parseInt(_r.transactionIndex.split(":")[0]) - 1
					var trIndex = parseInt(_r.transactionIndex.split(":")[1])
					if (blockIndex <= model.blocks.length - 1)
					{
						var item = model.blocks[blockIndex]
						if (trIndex <= item.transactions.length - 1)
						{
							var tr = item.transactions[trIndex]
							tr.returned = _r.returned
							tr.recordIndex = _r.recordIndex
							tr.logs = _r.logs
							tr.sender = _r.sender
							var trModel = blockModel.getTransaction(blockIndex, trIndex)
							trModel.returned = _r.returned
							trModel.recordIndex = _r.recordIndex
							trModel.logs = _r.logs
							trModel.sender = _r.sender
							blockModel.setTransaction(blockIndex, trIndex, trModel)
							return;
						}
					}
					// tr is not in the list.
					var itemTr = TransactionHelper.defaultTransaction()
					itemTr.saveStatus = false
					itemTr.functionId = _r.function
					itemTr.contractId = _r.contract
					itemTr.gasAuto = true
					itemTr.parameters = _r.parameters
					itemTr.isContractCreation = itemTr.functionId === itemTr.contractId
					itemTr.label = _r.label
					itemTr.isFunctionCall = itemTr.functionId !== ""
					itemTr.returned = _r.returned
					itemTr.value = QEtherHelper.createEther(_r.value, QEther.Wei)
					itemTr.sender = _r.sender
					itemTr.recordIndex = _r.recordIndex
					itemTr.logs = _r.logs
					model.blocks[model.blocks.length - 1].transactions.push(itemTr)
					blockModel.appendTransaction(itemTr)
				}
				onMiningComplete:
				{
				}
			}

			ScenarioButton {
				id: newAccount
				text: qsTr("New Account..")
				onClicked: {
					model.accounts.push(projectModel.stateListModel.newAccount("1000000", QEther.Ether))
				}
				Layout.preferredWidth: 100
				Layout.preferredHeight: 30
				buttonShortcut: ""
				sourceImg: "qrc:/qml/img/newaccounticon@2x.png"
				roundLeft: true
				roundRight: true
			}
		}
	}

	TransactionDialog {
		id: transactionDialog
		property bool execute
		onAccepted: {
			var item = transactionDialog.getItem()
			if (execute)
			{
				var lastBlock = model.blocks[model.blocks.length - 1];
				if (lastBlock.status === "mined")
				{
					var newBlock = projectModel.stateListModel.createEmptyBlock();
					model.blocks.push(newBlock);
					blockModel.appendBlock(newBlock)
				}
				if (!clientModel.running)
					clientModel.executeTr(item)
			}
			else {
				model.blocks[blockIndex].transactions[transactionIndex] = item
				blockModel.setTransaction(blockIndex, transactionIndex, item)
				chainChanged()
			}

		}
	}
}


