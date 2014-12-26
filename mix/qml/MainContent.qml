import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1
import CodeEditorExtensionManager 1.0

Rectangle {
	objectName: "mainContent"
	signal keyPressed(variant event)
	focus: true
	Keys.enabled: true
	Keys.onPressed:
	{
		root.keyPressed(event.key);
	}
	anchors.fill: parent
	height: parent.height
	width: parent.width;
	id:root

	CodeEditorExtensionManager {
		headerView: headerView;
		rightView: rightView;
		editor: codeEditor
	}

	Column {
		anchors.fill: parent;
		anchors.horizontalCenter: parent.horizontalCenter
		anchors.verticalCenter: parent.verticalCenter
		Rectangle {
			id: headerView
			height: 50
			width: parent.width;
		}
		Row {
			anchors.fill: parent;
			anchors.horizontalCenter: parent.horizontalCenter
			anchors.verticalCenter: parent.verticalCenter
			SplitView {
				orientation: Qt.Vertical;
				anchors.fill: parent;
				Rectangle {
					id: editorRect;
					height: parent.height;
					width: parent.width /2;
					TextArea {
						id: codeEditor
						height: parent.height
						width: parent.width
						font.family: "Monospace"
						font.pointSize: 12
						anchors.centerIn: parent
						tabChangesFocus: false
						Keys.onPressed: {
								if (event.key === Qt.Key_Tab) {
									codeEditor.insert(codeEditor.cursorPosition, "\t");
									event.accepted = true;
								}
							}
					}
				}
				Rectangle {
					id: rightView;
					height: parent.height;
					width: parent.width /2;
					Rectangle {
						id: debugWindow;
					}
					Rectangle {
						anchors.right: parent.right
						id: rightPaneView
						width: parent.width * 0.2
						height: parent.height
						Layout.minimumWidth: 20
						TabView {
							id: rightPaneTabs
							antialiasing: true
							anchors.fill: parent
						}
					}
				}
			}
		}
	}
}
