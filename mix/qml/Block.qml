import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import Qt.labs.settings 1.0
import "js/Debugger.js" as Debugger
import "js/ErrorLocationFormater.js" as ErrorLocationFormater
import "."

ColumnLayout
{
	id: root
	property variant transactions
	property string status
	property int number
	property int blockWidth: Layout.preferredWidth - statusWidth - horizontalMargin
	property int horizontalMargin: 10
	property int trHeight: 35
	spacing: 0
	property int openedTr: 0
	property int blockIndex
	property variant scenario
	property string labelColor: "#414141"
	property int scenarioIndex
	signal txSelected(var txIndex)

	function calculateHeight()
	{
		if (transactions)
		{
			if (index >= 0)
				return trHeight + trHeight * transactions.count + openedTr
			else
				return trHeight
		}
		else
			return trHeight
	}

	function editTx(txIndex)
	{
		transactionDialog.stateAccounts = scenario.accounts
		transactionDialog.execute = false
		transactionDialog.editMode = true
		transactionDialog.open(txIndex, blockIndex,  transactions.get(txIndex))
	}

	function select(txIndex)
	{
		transactionRepeater.itemAt(txIndex).select()
	}

	onOpenedTrChanged:
	{
		Layout.preferredHeight = calculateHeight()
		height = calculateHeight()
	}

	DebuggerPaneStyle {
		id: dbgStyle
	}

	Rectangle
	{
		id: top
		Layout.preferredWidth: blockWidth
		height: 10
		anchors.bottom: rowHeader.top
		color: "#DEDCDC"
		radius: 15
		anchors.left: parent.left
		anchors.leftMargin: statusWidth
		anchors.bottomMargin: -5
	}

	RowLayout
	{
		Layout.preferredHeight: trHeight
		Layout.preferredWidth: blockWidth
		id: rowHeader
		spacing: 0
		Rectangle
		{
			Layout.preferredWidth: blockWidth
			Layout.preferredHeight: trHeight
			color: "#DEDCDC"
			anchors.left: parent.left
			anchors.leftMargin: statusWidth
			Label {
				anchors.verticalCenter: parent.verticalCenter
				anchors.left: parent.left
				anchors.leftMargin: horizontalMargin
				font.pointSize: dbgStyle.absoluteSize(1)
				color: "#adadad"
				text:
				{
					if (number === -2)
						return qsTr("GENESIS PARAMETERS")
					else if (status === "mined")
						return qsTr("BLOCK") + " " + number
					else
						return qsTr("PENDING TRANSACTIONS")
				}
			}

			Label
			{
				text: qsTr("EDIT")
				color:  "#1397da"
				anchors.verticalCenter: parent.verticalCenter
				anchors.right: parent.right
				anchors.rightMargin: 14
				visible: number === -2
				MouseArea
				{
					anchors.fill: parent
					onClicked:
					{
						// load edit block panel
						projectModel.stateListModel.editState(scenarioIndex)
					}
				}
			}
		}
	}

	Repeater // List of transactions
	{
		id: transactionRepeater
		model: transactions
		RowLayout
		{
			id: rowTransaction
			Layout.preferredHeight: trHeight
			spacing: 0

			function select()
			{
				rowContentTr.select()
			}

			function displayContent()
			{
				logsText.text = ""
				if (index >= 0 && transactions.get(index).logs && transactions.get(index).logs.count)
				{
					for (var k = 0; k < transactions.get(index).logs.count; k++)
					{
						var log = transactions.get(index).logs.get(k)
						if (log.name)
							logsText.text += log.name + ":\n"
						else
							logsText.text += "log:\n"

						if (log.param)
							for (var i = 0; i < log.param.count; i++)
							{
								var p = log.param.get(i)
								logsText.text += p.name + " = " + p.value + " - indexed:" + p.indexed + "\n"
							}
						else {
							logsText.text += "From : " + log.address + "\n"
						}
					}
					logsText.text += "\n\n"
				}
				rowDetailedContent.visible = !rowDetailedContent.visible
			}

			Rectangle
			{
				id: trSaveStatus
				Layout.preferredWidth: statusWidth
				Layout.preferredHeight: parent.height
				color: "transparent"
				anchors.top: parent.top
				property bool saveStatus
				Image {
					anchors.top: parent.top
					anchors.left: parent.left
					anchors.leftMargin: -4
					anchors.topMargin: 0
					id: saveStatusImage
					source: "qrc:/qml/img/recyclediscard@2x.png"
					width: statusWidth + 10
					fillMode: Image.PreserveAspectFit
				}

				Component.onCompleted:
				{
					if (index >= 0)
						saveStatus = transactions.get(index).saveStatus
				}

				onSaveStatusChanged:
				{
					if (saveStatus)
						saveStatusImage.source = "qrc:/qml/img/recyclekeep@2x.png"
					else
						saveStatusImage.source = "qrc:/qml/img/recyclediscard@2x.png"

					if (index >= 0)
						transactions.get(index).saveStatus = saveStatus
				}

				MouseArea {
					id: statusMouseArea
					anchors.fill: parent
					onClicked:
					{
						parent.saveStatus = !parent.saveStatus
					}
				}
			}

			Rectangle
			{
				Layout.preferredWidth: blockWidth
				Layout.preferredHeight: trHeight
				height: trHeight
				color: "#DEDCDC"
				id: rowContentTr
				anchors.top: parent.top

				property bool selected: false
				Connections
				{
					target: blockChainPanel
					onTxSelected: {
						if (root.blockIndex !== blockIndex || index !== txIndex)
							rowContentTr.deselect()
					}
				}

				function select()
				{
					rowContentTr.selected = true
					rowContentTr.color = "#4F4F4F"
					hash.color = "#EAB920"
					func.color = "#EAB920"
					txSelected(index)
				}

				function deselect()
				{
					rowContentTr.selected = false
					rowContentTr.color = "#DEDCDC"
					hash.color = labelColor
					func.color = labelColor
				}

				MouseArea
				{
					anchors.fill: parent
					onClicked: {
						if (!rowContentTr.selected)
							rowContentTr.select()
						else
							rowContentTr.deselect()

					}
					onDoubleClicked:
					{
						root.editTx(index)
					}
				}

				RowLayout
				{
					Layout.fillWidth: true
					Layout.preferredHeight: trHeight - 10
					anchors.verticalCenter: parent.verticalCenter
					Rectangle
					{
						Layout.preferredWidth: fromWidth
						anchors.left: parent.left
						anchors.leftMargin: horizontalMargin
						Text
						{
							id: hash
							width: parent.width - 30
							elide: Text.ElideRight
							anchors.verticalCenter: parent.verticalCenter
							maximumLineCount: 1
							color: labelColor
							font.pointSize: dbgStyle.absoluteSize(1)
							font.bold: true
							text: {
								if (index >= 0)
									return clientModel.resolveAddress(transactions.get(index).sender)
								else
									return ""
							}
						}
					}

					Rectangle
					{
						Layout.preferredWidth: toWidth
						Text
						{
							id: func
							text: {
								if (index >= 0)
									parent.parent.userFrienldyToken(transactions.get(index).label)
								else
									return ""
							}
							elide: Text.ElideRight
							anchors.verticalCenter: parent.verticalCenter
							color: labelColor
							font.pointSize: dbgStyle.absoluteSize(1)
							font.bold: true
							maximumLineCount: 1
							width: parent.width
						}
					}

					function userFrienldyToken(value)
					{
						if (value && value.indexOf("<") === 0)
						{
							if (value.split("> ")[1] === " - ")
								return value.split(" - ")[0].replace("<", "")
							else
								return value.split(" - ")[0].replace("<", "") + "." + value.split("> ")[1] + "()";
						}
						else
							return value
					}					
				}
			}

			Rectangle
			{
				width: debugActionWidth
				height: trHeight - 10
				anchors.right: rowContentTr.right
				anchors.top: rowContentTr.top
				anchors.rightMargin: 10
				color: "transparent"

				Image {
					id: debugImg
					source: "qrc:/qml/img/rightarrow@2x.png"
					width: debugActionWidth
					fillMode: Image.PreserveAspectFit
					anchors.horizontalCenter: parent.horizontalCenter
					visible: transactions.get(index).recordIndex !== undefined
				}
				MouseArea
				{
					anchors.fill: parent
					onClicked:
					{
						if (transactions.get(index).recordIndex !== undefined)
						{
							debugTrRequested = [ blockIndex, index ]
							clientModel.debugRecord(transactions.get(index).recordIndex);
						}
					}
				}
			}
		}
	}
}

