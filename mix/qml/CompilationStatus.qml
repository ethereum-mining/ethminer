import QtQuick 2.2
import QtQuick.Controls 1.1

Rectangle {
	anchors.fill: parent
	width: parent.width
	height: parent.height
	color: "lightgray"
	Text {
		font.pointSize: 9
		anchors.left: parent.left
		anchors.top: parent.top
		anchors.topMargin: 3
		anchors.leftMargin: 3
		height: 9
		font.family: "Monospace"
		objectName: "status"
		id: status
	}
	TextArea {
		readOnly: true
		anchors.left: parent.left
		anchors.leftMargin: 10
		anchors.top: status.bottom
		anchors.topMargin: 3
		font.pointSize: 9
		font.family: "Monospace"
		height: parent.height * 0.8
		width:  parent.width - 20
		wrapMode: Text.Wrap
		backgroundVisible: false
		objectName: "content"
		id: content
	}
}
