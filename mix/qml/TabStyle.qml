import QtQuick 2.3
import QtQuick.Controls 1.2
import QtQuick.Controls.Styles 1.2

TabViewStyle {
	frameOverlap: 1
	tabBar: Rectangle {
		color: "lightgray"
	}
	tab: Rectangle {
		color: "lightsteelblue"
		implicitWidth: Math.max(text.width + 4, 80)
		implicitHeight: 20
		radius: 2
		Text {
			id: text
			anchors.centerIn: parent
			text: styleData.title
			color: styleData.selected ? "white" : "black"
		}
	}
	frame: Rectangle { color: "steelblue" }
}
