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
    SplitView {
        orientation: Qt.Horizontal
        anchors.fill: parent
        SplitView {
            //anchors.fill: parent
            width: parent.width * 0.8
            orientation: Qt.Vertical
            Rectangle {
                anchors.top: parent.top
                id: contentView
                width: parent.width
                height: parent.height * 0.7

				Item {
						anchors.fill: parent
						Rectangle {
							id: lineColumn
							property int rowHeight: codeEditor.font.pixelSize + 3
							color: "#202020"
							width: 50
							height: parent.height
							Column {
								y: -codeEditor.flickableItem.contentY + 4
								width: parent.width
								Repeater {
									model: Math.max(codeEditor.lineCount + 2, (lineColumn.height/lineColumn.rowHeight))
									delegate: Text {
										id: text
										color: codeEditor.textColor
										font: codeEditor.font
										width: lineColumn.width - 4
										horizontalAlignment: Text.AlignRight
										verticalAlignment: Text.AlignVCenter
										height: lineColumn.rowHeight
										renderType: Text.NativeRendering
										text: index + 1
									}
								}
							}
						}

					TextArea {
						id: codeEditor
						textColor: "#EEE8D5"
						style: TextAreaStyle {
							backgroundColor: "#002B36"
						}

						anchors.left: lineColumn.right
						anchors.right: parent.right
						anchors.top: parent.top
						anchors.bottom: parent.bottom
						wrapMode: TextEdit.NoWrap
						frameVisible: false

						height: parent.height
						font.family: "Monospace"
						font.pointSize: 12
						width: parent.width
						//anchors.centerIn: parent
						tabChangesFocus: false
						Keys.onPressed: {
							if (event.key === Qt.Key_Tab) {
								codeEditor.insert(codeEditor.cursorPosition, "\t");
								event.accepted = true;
							}
						}
					}
				}

            }
            Rectangle {
                anchors.bottom: parent.bottom
                id: contextualView
                width: parent.width
                Layout.minimumHeight: 20
                height: parent.height * 0.3
                TabView {
                    id: contextualTabs
                    antialiasing: true
                    anchors.fill: parent
                    style: TabStyle {}
                }
            }
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
                //style: TabStyle {}
            }
        }
        CodeEditorExtensionManager {
                tabView: contextualTabs
                rightTabView: rightPaneTabs
                editor: codeEditor
        }
    }
}
