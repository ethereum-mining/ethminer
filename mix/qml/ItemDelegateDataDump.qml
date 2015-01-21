import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1

Rectangle {
	anchors.fill: parent
	RowLayout
	{
		id: row;
		anchors.fill: parent
		spacing: 2
		Rectangle
		{
			id: firstCol;
			color: "#f7f7f7"
			Layout.fillWidth: true
			Layout.minimumWidth: 35
			Layout.preferredWidth: 35
			Layout.maximumWidth: 35
			Layout.minimumHeight: parent.height
			Text {
				anchors.centerIn: parent
				font.family: "monospace"
				font.bold: true
				color: "#4a4a4a"
				text: modelData[0]
				font.pointSize: 9;
			}
		}

		Rectangle
		{
			Layout.fillWidth: true
			Layout.minimumWidth: 90
			Layout.preferredWidth: 90
			Layout.maximumWidth: 90
			Layout.minimumHeight: parent.height
			Text {
				font.family: "monospace"
				font.bold: true
				anchors.verticalCenter: parent.verticalCenter
				color: "#4a4a4a"
				text: modelData[1]
				font.pointSize: 9
			}
		}

		Rectangle
		{
			Layout.fillWidth: true
			Layout.minimumWidth: 50
			Layout.minimumHeight: parent.height
			Text {
				anchors.verticalCenter: parent.verticalCenter
				anchors.horizontalCenter:  parent.horizontalCenter
				font.family: "monospace"
				color: "#4a4a4a"
				text: modelData[2]
				font.pointSize: 9
			}
		}
	}

	Rectangle {
		width: parent.width;
		height: 1;
		color: "#cccccc"
		anchors.bottom: parent.bottom
	}
}
