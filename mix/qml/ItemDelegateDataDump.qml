import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1
import "."

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
				anchors.verticalCenter: parent.verticalCenter
				anchors.left: parent.left
				anchors.leftMargin: 3
				font.family: "monospace"
				font.bold: true
				color: "#4a4a4a"
				text: modelData[0]
				font.pointSize: DebuggerPaneStyle.general.dataDumpFontSize;
			}
		}

		Rectangle
		{
			Layout.fillWidth: true
			Layout.minimumWidth: 110
			Layout.preferredWidth: 110
			Layout.maximumWidth: 110
			Layout.minimumHeight: parent.height
			Text {
				font.family: "monospace"
				font.bold: true
				anchors.verticalCenter: parent.verticalCenter
				anchors.left: parent.left
				anchors.leftMargin: 4
				color: "#4a4a4a"
				text: modelData[1]
				font.pointSize: DebuggerPaneStyle.general.dataDumpFontSize
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
