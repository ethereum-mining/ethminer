import QtQuick 2.2
import QtQuick.Controls.Styles 1.1
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1

Item {
	property alias model: callTable.model
	signal frameActivated(int index)
	ColumnLayout {
		anchors.fill: parent
		Text {
			text: qsTr("Call Stack")
			Layout.fillWidth: true
		}
		TableView {
			id: callTable
			Layout.fillWidth: true
			Layout.fillHeight: true
			headerDelegate: null

			TableViewColumn {
				role: "modelData"
				title: qsTr("Address")
				width: parent.width
			}
			onActivated:  {
				frameActivated(row);
			}
		}
	}
}
