import QtQuick 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.3

Rectangle {
	anchors.fill: parent
	color: "white"
	property variant sel
	signal selected(string step)

	function init()
	{
		menu.itemAt(0).select()
	}

	function itemClicked(step)
	{
		selected(step)
	}

	border.color: "#cccccc"
	border.width: 1

	Column
	{
		anchors.fill: parent
		anchors.margins: 1
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
					label: qsTr("Package files")
				},
				{
					step: 3,
					type:"register",
					label: qsTr("Register Dapp")
				}
			]

			Rectangle
			{
				height: 50
				width: parent.width
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

	}
}

