import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import Qt.labs.settings 1.0
import "js/Debugger.js" as Debugger
import "js/ErrorLocationFormater.js" as ErrorLocationFormater
import "."


Rectangle {
	color: "#ededed"
	property alias bc: blockChain

	Connections
	{
		target:  projectModel
		onProjectLoaded: {
			loader.init()
		}
	}

	ScrollView
	{
		anchors.fill: parent
		onWidthChanged: {
			columnExe.width = width - 40
		}

		ColumnLayout
		{
			id: columnExe
			Layout.preferredWidth: parent.width
			anchors.left: parent.left
			anchors.leftMargin: 15
			ColumnLayout
			{				
				id: scenarioColumn
				width: parent.width
				spacing: 10				
				ScenarioLoader
				{
					anchors.horizontalCenter: parent.horizontalCenter
					height: 100
					Layout.preferredWidth: 400
					width: 400
					id: loader
				}

				Connections
				{
					target: blockChain
					onChainChanged:
					{
						loader.needSaveOrReload()
					}
				}

				Rectangle
				{
					Layout.preferredWidth: parent.width
					height: 1
					color: "#cccccc"
				}

				Connections
				{
					target: loader
					onLoaded:
					{
						watchers.clear()
						blockChain.load(scenario, loader.selectedScenarioIndex)
					}
				}

				BlockChain
				{
					id: blockChain
					width: parent.width
				}

				Connections
				{
					target: blockChain
					property var currentSelectedBlock
					property var currentSelectedTx
					onTxSelected:
					{
						currentSelectedBlock = blockIndex
						currentSelectedTx = txIndex
						updateWatchers(blockIndex, txIndex)
					}

					function updateWatchers(blockIndex, txIndex)
					{
						var tx = blockChain.model.blocks[blockIndex].transactions[txIndex]
						var state = blockChain.getState(tx.recordIndex)
						watchers.updateWidthTx(tx, state, blockIndex, txIndex)
					}

					onRebuilding: {
						watchers.clear()
					}

					onAccountAdded: {
						watchers.addAccount(address, "0 wei")
					}
				}
			}

			Watchers
			{
				id: watchers
				bc: blockChain
				Layout.fillWidth: true
				Layout.preferredHeight: 740
			}

			Rectangle
			{
				color: "transparent"
				Layout.preferredHeight: 50
				Layout.fillWidth: true
			}
		}
	}
}
