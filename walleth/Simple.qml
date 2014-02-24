import QtQuick 2.1
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0

Item {
	id: main
	anchors.fill: parent
	anchors.margins: 9
	Label {
		text: "This fills the whole cell"
		Layout.minimumHeight: 30
		Layout.fillHeight: true
		Layout.fillWidth: true
	}


}
