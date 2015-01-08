import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1

ColumnLayout {
	property string currentStep
	property string mem
	property string stepCost
	property string gasSpent
	spacing: 0
	RowLayout {
		width: parent.width
		height: parent.height / 4
		Rectangle {
			width: parent.width / 2
			height: parent.height
			color: "#e5e5e5"
			Text
			{
				font.pixelSize: 12
				anchors.centerIn: parent
				color: "#a2a2a2"
				text: qsTr("Current step")
				font.family: "Sans Serif"
			}
		}
		Text
		{
			font.pixelSize: 13
			id: currentStepValue
			text: currentStep
		}
	}

	RowLayout {
		width: parent.width
		height: parent.height / 4
		Rectangle {
			width: parent.width / 2
			height: parent.height
			color: "#e5e5e5"
			Text
			{
				font.pixelSize: 12
				anchors.centerIn: parent
				color: "#a2a2a2"
				text: qsTr("Adding memory")
				font.family: "Sans Serif"
			}
		}

		Text
		{
			font.pixelSize: 13
			id: memValue
			text: mem
		}
	}

	RowLayout {
		width: parent.width
		height: parent.height / 4
		Rectangle {
			width: parent.width / 2
			height: parent.height
			color: "#e5e5e5"
			Text
			{
				font.pixelSize: 12
				anchors.centerIn: parent
				color: "#a2a2a2"
				text: qsTr("Step cost")
				font.family: "Sans Serif"
			}
		}
		Text
		{
			font.pixelSize: 13
			id: stepCostValue
			text: stepCost
		}
	}

	RowLayout {
		width: parent.width
		height: parent.height / 4
		Rectangle {
			width: parent.width / 2
			height: parent.height
			color: "#e5e5e5"
			Text
			{
				font.pixelSize: 12
				anchors.centerIn: parent
				color: "#a2a2a2"
				text: qsTr("Total gas spent")
				font.family: "Sans Serif"
			}
		}
		Text
		{
			font.pixelSize: 13
			id: gasSpentValue
			text: gasSpent
		}
	}
}

