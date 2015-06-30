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

	Column
	{
		id: scenarioColumn
		anchors.margins: 10
		anchors.fill: parent
		spacing: 10
		ScenarioLoader
		{
			height: 100
			width: parent.width
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
			width: parent.parent.width
			height: 1
			color: "#cccccc"
		}

		Connections
		{
			target: loader
			onLoaded:
			{
				blockChain.load(scenario)
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
			onTxSelected:{
				var tx = model.block[blockIndex].transactions[txIndex]
				var state =  blockChain.getState(tx.recordIndex)
				watchers.updateWidthTx(tx, state)
			}
		}
	}

	ScrollView
	{
		anchors.top: scenarioColumn.bottom
		width: parent.width
		height: 500
		Watchers
		{
			id: watchers
			anchors.topMargin: 10
		}
	}
}
