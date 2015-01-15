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
				anchors.leftMargin: 5
				color: "#8b8b8b"
				text: modelData[0]
				font.pointSize: 9;
			}
		}

		Rectangle
		{
			anchors.left: firstCol.right
			Layout.fillWidth: true
			Layout.minimumWidth: 90
			Layout.preferredWidth: 90
			Layout.maximumWidth: 90
			Layout.minimumHeight: parent.height
			Text {
				anchors.left: parent.left
				anchors.leftMargin: 7
				anchors.verticalCenter: parent.verticalCenter
				color: "#8b8b8b"
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
				anchors.left: parent.left
				anchors.verticalCenter: parent.verticalCenter
				color: "#ededed"
				font.bold: true
				text: modelData[2]
				font.pointSize: 10
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
