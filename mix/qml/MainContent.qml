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

	function hideRightView()
	{
		if (rightView.visible)
		{
			rightView.hide();
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

				Keys.onEscapePressed:
				{
					hide();
				}

				visible: false;
				id: rightView;
				property real panelRelWidth: 0.38
				function show() {
					visible = true;
					editorRect.width = parent.width * (1 - 0.38)
				}

				function hide() {
					visible = false;
					editorRect.width = parent.width;
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




