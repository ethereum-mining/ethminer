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
	id: root

	function ensureRightView()
	{
		if (!rightView.visible)
		{
			rightView.show();
		}
	}

	CodeEditorExtensionManager {
		headerView: headerPaneTabs;
		rightView: rightPaneTabs;
		editor: codeEditor
	}

	GridLayout
	{
		anchors.fill: parent
		rows: 2
		flow: GridLayout.TopToBottom
		columnSpacing: 0
		rowSpacing: 0
		Rectangle {
			Layout.row: 0
			Layout.fillWidth: true
			Layout.preferredHeight: 50
			id: headerView
			TabView {
				id: headerPaneTabs
				tabsVisible: false
				antialiasing: true
				anchors.fill: parent
				style: TabViewStyle {
					frameOverlap: 1
					tab: Rectangle {}
					frame: Rectangle {}
				}
			}
		}

		SplitView {
			resizing: false
			Layout.row: 1
			orientation: Qt.Horizontal;
			Layout.fillWidth: true
			Layout.preferredHeight: root.height - headerView.height;
			Rectangle {
				id: editorRect;
				height: parent.height;
				width: parent.width;
				TextArea {
					id: codeEditor
					anchors.fill: parent;
					font.family: "Monospace"
					font.pointSize: 12
					backgroundVisible: true;
					textColor: "white"
					tabChangesFocus: false
					style: TextAreaStyle {
							backgroundColor: "black"
						}
					Keys.onPressed: {
							if (event.key === Qt.Key_Tab) {
								codeEditor.insert(codeEditor.cursorPosition, "\t");
								event.accepted = true;
							}
						}
				}
			}
			Rectangle {
				visible: false;
				id: rightView;
				property real panelRelWidth: 0.38
				function show() {
					visible = true;
					editorRect.width = parent.width * (1 - 0.38)
					codeEditor.focus = false;
					rightPaneTabs.focus = true;
				}
				height: parent.height;
				width: Layout.minimumWidth
				Layout.minimumWidth: parent.width * 0.38
				Rectangle {
					anchors.fill: parent;
					id: rightPaneView
					TabView {
						id: rightPaneTabs
						tabsVisible: false
						antialiasing: true
						anchors.fill: parent
						style: TabViewStyle {
							frameOverlap: 1
							tab: Rectangle {}
							frame: Rectangle {}
						}
					}
				}
			}
		}
	}
}
