import QtQuick 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.3

Rectangle {
	anchors.fill: parent
	color: "white"
	property var worker
	property variant sel
	signal selected(string step)
	id: root

	function refreshCurrent()
	{
		menu.itemAt(sel).select()
	}

	function init()
	{
		menu.itemAt(0).select()
	}

	function itemClicked(step)
	{
		selected(step)
	}

	function reset()
	{
		for (var k in deployLogs.logs)
		{
			deployLogs.logs[k] = ""
		}
		deployLogs.switchLogs()
		refreshCurrent()
	}

	border.color: "#cccccc"
	border.width: 1


	Connections
	{
		id: deployStatus
		target: deploymentDialog.deployStep
		onDeployed:
		{
			console.log("deployed")
		}
	}

	Connections
	{
		id: packagedStatus
		target: deploymentDialog.packageStep
		onPackaged:
		{
			console.log("packaged")
		}
	}

	Connections
	{
		id: registerStatus
		target: deploymentDialog.registerStep
		onRegistered:
		{
			console.log("registered")
		}
	}

	ColumnLayout
	{
		anchors.fill: parent
		anchors.margins: 1
		spacing: 0
		Repeater
		{
			id: menu
			model: [
				{
					step: 1,
					type:"deploy",
					label: qsTr("Deploy contracts")
				},
				{
					step: 2,
					type:"package",
					label: qsTr("Package Dapp")
				},
				{
					step: 3,
					type:"register",
					label: qsTr("Register Dapp")
				}
			]

			Rectangle
			{
				Layout.preferredHeight: 50
				Layout.fillWidth: true
				color: "white"
				id: itemContainer

				function select()
				{
					if (sel !== undefined)
					{
						menu.itemAt(sel).unselect()
					}
					labelContainer.state = "selected"
					sel = index
					itemClicked(menu.model[index].type)
					deployLogs.switchLogs()
				}

				function unselect()
				{
					labelContainer.state = ""
				}

				Rectangle {
					width: 40
					height: 40
					color: "transparent"
					border.color: "#cccccc"
					border.width: 2
					radius: width*0.5
					anchors.verticalCenter: parent.verticalCenter
					anchors.left: parent.left
					anchors.leftMargin: 10
					id: labelContainer
					Label
					{
						color: "#cccccc"
						id: label
						anchors.centerIn: parent
						text: menu.model[index].step
					}
					states: [
						State {
							name: "selected"
							PropertyChanges { target: label; color: "white" }
							PropertyChanges { target: labelContainer.border; color: "white" }
							PropertyChanges { target: detail; color: "white" }
							PropertyChanges { target: itemContainer; color: "#3395FE" }
						}
					]
				}

				Rectangle
				{
					anchors.verticalCenter: parent.verticalCenter
					anchors.left: label.parent.right
					width: parent.width - 40
					height: 40
					color: "transparent"
					Label
					{
						id: detail
						color: "black"
						anchors.verticalCenter: parent.verticalCenter
						anchors.left: parent.left
						anchors.leftMargin: 10
						text: menu.model[index].label
					}
				}

				MouseArea
				{
					anchors.fill: parent
					onClicked:
					{
						itemContainer.select()
					}
				}
			}
		}

		Connections {
			property var logs: ({})
			id: deployLogs

			function switchLogs()
			{
				if (root.sel)
				{
					if (!logs[root.sel])
						logs[root.sel] = ""
					log.text = logs[root.sel]
				}
			}

			target: projectModel
			onDeploymentStarted:
			{
				if (!logs[root.sel])
					logs[root.sel] = ""
				logs[root.sel] = logs[root.sel] + qsTr("Running deployment...") + "\n"
				log.text = logs[root.sel]
			}

			onDeploymentError:
			{
				if (!logs[root.sel])
					logs[root.sel] = ""
				logs[root.sel] = logs[root.sel] + error + "\n"
				log.text = logs[root.sel]
			}

			onDeploymentComplete:
			{
				if (!logs[root.sel])
					logs[root.sel] = ""
				logs[root.sel] = logs[root.sel] + qsTr("Deployment complete") + "\n"
				log.text = logs[root.sel]
			}

			onDeploymentStepChanged:
			{
				if (!logs[root.sel])
					logs[root.sel] = ""
				logs[root.sel] = logs[root.sel] + message + "\n"
				log.text = logs[root.sel]
			}
		}

		Rectangle
		{
			Layout.fillWidth: true
			Layout.preferredHeight: 2
			color: "#cccccc"
		}

		ScrollView
		{
			Layout.fillHeight: true
			Layout.fillWidth: true
			Text
			{
				anchors.left: parent.left
				anchors.leftMargin: 2
				font.pointSize: 9
				font.italic: true
				id: log
			}
		}

		Rectangle
		{
			Layout.preferredHeight: 20
			Layout.fillWidth: true
			color: "#cccccc"
			LogsPaneStyle
			{
				id: style
			}

			Label
			{
				anchors.horizontalCenter: parent.horizontalCenter
				anchors.verticalCenter: parent.verticalCenter
				text: qsTr("Logs")
				font.italic: true
				font.pointSize: style.absoluteSize(-1)
			}

			Button
			{
				height: 20
				width: 20
				anchors.right: parent.right
				action: clearAction
				iconSource: "qrc:/qml/img/cleariconactive.png"
				tooltip: qsTr("Clear Messages")
			}

			Action {
				id: clearAction
				enabled: log.text !== ""
				tooltip: qsTr("Clear")
				onTriggered: {
					deployLogs.logs[root.sel] = ""
					log.text = deployLogs.logs[root.sel]
				}
			}
		}

	}
}

