import QtQuick 2.2
import QtQuick.Controls.Styles 1.1
import QtQuick.Controls 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import "js/Debugger.js" as Debugger

Rectangle {
	anchors.fill: parent;
	color: "lightgrey"
	Component.onCompleted: Debugger.init();
	Rectangle {
		color: "transparent"
		id: headerInfo
		anchors.horizontalCenter: parent.horizontalCenter
		width: parent.width
		height: 60
		anchors.top: parent.top
		Column {
			width: parent.width
			height: parent.height
			Rectangle {
				color: "transparent"
				width: parent.width
				height: 30
				Label {
					anchors.centerIn: parent
					font.family: "Verdana"
					font.pointSize: 9
					font.italic: true
					id: headerInfoLabel
				}
			}
			Rectangle {
				color: "transparent"
				width: parent.width
				anchors.horizontalCenter: parent.horizontalCenter
				height: 30
				ListView {
					orientation: ListView.Horizontal
					anchors.centerIn: parent;
					width: parent.width
					id: headerReturnList
					delegate: renderDelegateReturnValues
				}
				Component {
					id: renderDelegateReturnValues
					Item {
						id: wrapperItem
						width: 80
						Text {
							anchors.centerIn: parent
							text: variable.declaration.name + " = " + variable.value
							font.pointSize: 9
						}
					}
				}
			}
		}
	}

	Keys.onPressed: {
		if (event.key === Qt.Key_F10)
			Debugger.moveSelection(1);
		else if (event.key === Qt.Key_F9)
			Debugger.moveSelection(-1);
	}

	Rectangle {
		color: "transparent"
		id: stateListContainer
		focus: true
		anchors.topMargin: 10
		anchors.top: headerInfo.bottom
		anchors.left: parent.left
		height: parent.height - 70
		width: parent.width * 0.5

		ListView {
			anchors.top: parent.top
			height: parent.height * 0.60
			width: 200
			anchors.horizontalCenter: parent.horizontalCenter
			id: statesList
			model: humanReadableExecutionCode
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
					text: line
					font.pointSize: 9
				}
			}
		}

		Rectangle {
			id: callStackPanel
			anchors.top: statesList.bottom
			height: parent.height * 0.35
			width: parent.width
			anchors.topMargin: 15
			color: "transparent"
			Label {
				id: callStackLabel
				anchors.bottomMargin: 10
				horizontalAlignment: "AlignHCenter"
				font.family: "Verdana"
				font.pointSize: 8
				font.letterSpacing: 2
				width: parent.width
				height: 15
				text: qsTr("callstack")
			}

			ListView {
				height: parent.height - 15
				width: 200
				anchors.top: callStackLabel.bottom
				anchors.horizontalCenter: parent.horizontalCenter
				id: levelList
				delegate: Component {
					Item {
						Text {
							font.family: "Verdana"
							font.pointSize: 8
							text: modelData
						}
					}
				}
			}
		}
	}

	Rectangle {
		color: "transparent"
		anchors.topMargin: 5
		anchors.bottomMargin: 10
		anchors.rightMargin: 10
		height: parent.height - 30
		width: parent.width * 0.5
		anchors.right: parent.right
		anchors.top: headerInfo.bottom
		anchors.bottom: parent.bottom

		Rectangle {
			id: debugStack
			anchors.top: parent.top
			width: parent.width
			height: parent.height * 0.25
			color: "transparent"
			Label {
				horizontalAlignment: "AlignHCenter"
				font.family: "Verdana"
				font.pointSize: 8
				width: parent.width
				height: 15
				anchors.top : parent.top
				text: qsTr("debug stack")
			}
			TextArea {
				anchors.bottom: parent.bottom
				width: parent.width
				font.family: "Verdana"
				font.pointSize: 8
				height: parent.height - 15
				id: debugStackTxt
				readOnly: true;
			}
		}

		Rectangle {
			id: debugMemory
			anchors.top: debugStack.bottom
			width: parent.width
			height: parent.height * 0.25
			color: "transparent"
			Label {
				horizontalAlignment: "AlignHCenter"
				font.family: "Verdana"
				font.pointSize: 8
				width: parent.width
				height: 15
				anchors.top : parent.top
				text: qsTr("debug memory")
			}
			TextArea {
				anchors.bottom: parent.bottom
				width: parent.width
				font.family: "Verdana"
				font.pointSize: 8
				height: parent.height - 15
				id: debugMemoryTxt
				readOnly: true;
			}
		}

		Rectangle {
			id: debugStorage
			anchors.top: debugMemory.bottom
			width: parent.width
			height: parent.height * 0.25
			color: "transparent"
			Label {
				horizontalAlignment: "AlignHCenter"
				font.family: "Verdana"
				font.pointSize: 8
				width: parent.width
				height: 15
				anchors.top : parent.top
				text: qsTr("debug storage")
			}
			TextArea {
				anchors.bottom: parent.bottom
				width: parent.width
				font.family: "Verdana"
				font.pointSize: 8
				height: parent.height - 15
				id: debugStorageTxt
				readOnly: true;
			}
		}

		Rectangle {
			id: debugCallData
			anchors.top: debugStorage.bottom
			width: parent.width
			height: parent.height * 0.25
			color: "transparent"
			Label {
				horizontalAlignment: "AlignHCenter"
				font.family: "Verdana"
				font.pointSize: 8
				width: parent.width
				height: 15
				anchors.top : parent.top
				text: qsTr("debug calldata")
			}
			TextArea {
				anchors.bottom: parent.bottom
				width: parent.width
				height: parent.height - 15
				font.family: "Verdana"
				font.pointSize: 8
				font.letterSpacing: 2
				id: debugCallDataTxt
				readOnly: true;
			}
		}
	}
}
