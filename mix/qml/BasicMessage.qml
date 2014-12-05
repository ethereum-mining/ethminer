import QtQuick 2.3
import QtQuick.Controls.Styles 1.2
import QtQuick.Controls 1.2
import QtQuick.Dialogs 1.2
import QtQuick.Layouts 1.1

Rectangle {
	anchors.fill: parent
	color: "lightgrey"
	Label
	{
		width: parent.width
		height: parent.height
		horizontalAlignment: "AlignHCenter"
		verticalAlignment: "AlignVCenter"
		objectName: "messageContent"
		id: messageTxt
		text: ""
	}
}
