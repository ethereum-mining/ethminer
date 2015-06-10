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
	property int trHeight: 30
	spacing: 0
	property int openedTr: 0
	property int blockIndex
	property variant scenario

	function calculateHeight()
	{
		if (transactions)
		{
			if (index >= 0)
				return 30 + 30 * transactions.count + openedTr
			else
				return 30
		}
		else
			return 30
	}

	onOpenedTrChanged:
	{
		Layout.preferredHeight = calculateHeight()
		height = calculateHeight()
	}



	RowLayout
	{
		Layout.preferredHeight: trHeight
		Layout.preferredWidth: blockWidth
		id: rowHeader
		Rectangle
		{
			color: "#DEDCDC"
			Layout.preferredWidth: blockWidth
			Layout.preferredHeight: trHeight
			radius: 4
			anchors.left: parent.left
			anchors.leftMargin: statusWidth + 5
			Label {
				anchors.verticalCenter: parent.verticalCenter
				anchors.left: parent.left
				anchors.leftMargin: horizontalMargin
				text:
				{
					if (status === "mined")
						return qsTr("BLOCK") + " " + number
					else
						return qsTr("BLOCK") + " pending"
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
						else{
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
				Layout.preferredHeight: trHeight
				color: "transparent"
				anchors.top: parent.top
				property bool saveStatus

				Image {
					id: saveStatusImage
					source: "qrc:/qml/img/recyclediscard@2x.png"
					width: statusWidth
					fillMode: Image.PreserveAspectFit
					anchors.verticalCenter: parent.verticalCenter
					anchors.horizontalCenter: parent.horizontalCenter
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
				Layout.preferredHeight: parent.height
				color: "#DEDCDC"
				id: rowContentTr
				anchors.top: parent.top
				ColumnLayout
				{
					anchors.top: parent.top
					spacing: 10
					RowLayout
					{
						anchors.top: parent.top
						anchors.verticalCenter: parent.verticalCenter
						spacing: cellSpacing
						Text
						{
							id: hash
							anchors.left: parent.left
							anchors.leftMargin: horizontalMargin
							Layout.preferredWidth: fromWidth
							elide: Text.ElideRight
							maximumLineCount: 1
							text: {
								if (index >= 0)
									return transactions.get(index).sender
								else
									return ""
							}
						}

						Text
						{
							id: func
							text: {
								if (index >= 0)
									parent.userFrienldyToken(transactions.get(index).label)
								else
									return ""
							}
							elide: Text.ElideRight
							maximumLineCount: 1
							Layout.preferredWidth: toWidth
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

						Text
						{
							id: returnValue
							elide: Text.ElideRight
							maximumLineCount: 1
							Layout.preferredWidth: valueWidth
							text: {
								if (index >= 0 && transactions.get(index).returned)
									return transactions.get(index).returned
								else
									return ""
							}
						}

						Rectangle
						{
							Layout.preferredWidth: logsWidth
							Layout.preferredHeight: trHeight - 10
							width: logsWidth
							color: "transparent"
							Text
							{
								id: logs
								anchors.left: parent.left
								anchors.leftMargin: 10
								text: {
									if (index >= 0 && transactions.get(index).logs && transactions.get(index).logs.count)
										return transactions.get(index).logs.count
									else
										return ""
								}
							}
							MouseArea {
								anchors.fill: parent
								onClicked: {
									rowTransaction.displayContent();
								}
							}
						}

						Rectangle
						{
							Layout.preferredWidth: debugActionWidth
							Layout.preferredHeight: trHeight - 10
							color: "transparent"

							Image {
								source: "qrc:/qml/img/edit.png"
								width: 18
								fillMode: Image.PreserveAspectFit
								anchors.verticalCenter: parent.verticalCenter
								anchors.horizontalCenter: parent.horizontalCenter
							}
							MouseArea
							{
								anchors.fill: parent
								onClicked:
								{
									transactionDialog.stateAccounts = scenario.accounts
									transactionDialog.execute = false
									transactionDialog.open(index, blockIndex,  transactions.get(index))
								}
							}
						}

						Rectangle
						{
							Layout.preferredWidth: debugActionWidth
							Layout.preferredHeight: trHeight - 10
							color: "transparent"

							Image {
								id: debugImg
								source: "qrc:/qml/img/rightarrow@2x.png"
								width: statusWidth
								fillMode: Image.PreserveAspectFit
								anchors.verticalCenter: parent.verticalCenter
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

					RowLayout
					{
						id: rowDetailedContent
						visible: false
						Layout.preferredHeight:{
							if (index >= 0 && transactions.get(index).logs)
								return 100 * transactions.get(index).logs.count
							else
								return 100
						}
						onVisibleChanged:
						{
							var lognb = transactions.get(index).logs.count
							if (visible)
							{
								rowContentTr.Layout.preferredHeight = trHeight + 100 * lognb
								openedTr += 100 * lognb
							}
							else
							{
								rowContentTr.Layout.preferredHeight = trHeight
								openedTr -= 100 * lognb
							}
						}

						Text {
							anchors.left: parent.left
							anchors.leftMargin: horizontalMargin
							id: logsText
						}
					}
				}
			}
		}
	}
}

