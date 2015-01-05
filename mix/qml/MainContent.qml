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
                TextArea {
                    id: codeEditor
                    height: parent.height
					font.family: "Monospace"
					font.pointSize: 12
                    width: parent.width
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
