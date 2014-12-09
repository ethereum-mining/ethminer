import QtQuick 2.2
import QtQuick.Controls.Styles 1.2
import QtQuick.Controls 1.2
import QtQuick.Dialogs 1.2
import QtQuick.Layouts 1.1


Rectangle {
	color: "transparent"
	id: transactionListContainer
	focus: true
	anchors.topMargin: 10
	anchors.left: parent.left
	height: parent.height - 30
	width: parent.width * 0.5

	ListView {
		anchors.top: parent.top
		height: parent.height
		width: parent.width
		anchors.horizontalCenter: parent.horizontalCenter
		id: statesList
		model: transactionListModel
		delegate: renderDelegate
		highlight: highlightBar
		highlightFollowsCurrentItem: true
	}

	Component {
		id: highlightBar
		Rectangle {
			height: statesList.currentItem.height
			width: statesList.currentItem.width
			border.color: "orange"
			border.width: 1
			Behavior on y { SpringAnimation { spring: 2; damping: 0.1 } }
		}
	}

	Component {
		id: renderDelegate
		Item {
			id: wrapperItem
			height: 20
			width: parent.width
			Text {
				anchors.centerIn: parent
				text: title
				font.pointSize: 9
			}
		}
	}
}
