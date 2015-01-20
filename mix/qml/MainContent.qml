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

	function toggleRightView()
	{
		if (!rightView.visible)
			rightView.show();
		else
			rightView.hide();
	}

	function ensureRightView()
	{
		if (!rightView.visible)
			rightView.show();
	}

	function hideRightView()
	{
		if (rightView.visible)
			rightView.hide();
	}

	CodeEditorExtensionManager {
		headerView: headerPaneTabs;
		rightView: rightPaneTabs;
	}

	GridLayout
	{
		anchors.fill: parent
		rows: 2
		flow: GridLayout.TopToBottom
		columnSpacing: 0
		rowSpacing: 0
		Rectangle {
			width: parent.width
			height: 50
			Layout.row: 0
			Layout.fillWidth: true
			Layout.preferredHeight: 50
			id: headerView
			Rectangle
			{
				gradient: Gradient {
					GradientStop { position: 0.0; color: "#f1f1f1" }
					GradientStop { position: 1.0; color: "#d9d7da" }
				}
				id: headerPaneContainer
				anchors.fill: parent
				TabView {
					id: headerPaneTabs
					tabsVisible: false
					antialiasing: true
					anchors.fill: parent
					style: TabViewStyle {
						frameOverlap: 1
						tab: Rectangle {}
						frame: Rectangle { color: "transparent" }
					}
				}
			}
		}

		SplitView {
			resizing: false
			Layout.row: 1
			orientation: Qt.Horizontal;
			Layout.fillWidth: true
			Layout.preferredHeight: root.height - headerView.height;

			ProjectList	{
				id: projectList
				width: 200
				height: parent.height
				Layout.minimumWidth: 200
			}

			Rectangle {
				id: contentView
				width: parent.width - projectList.width
				height: parent.height
				SplitView {
					anchors.fill: parent
					orientation: Qt.Vertical
					CodeEditorView {
						height: parent.height * 0.6
						anchors.top: parent.top
						width: parent.width
					}
					WebPreview {
						height: parent.height * 0.4
						width: parent.width
					}
				}
			}

			Rectangle {
				visible: false;
				id: rightView;

				Keys.onEscapePressed:
				{
					hide();
				}

				function show() {
					visible = true;
					contentView.width = parent.width - projectList.width - rightView.width;
				}

				function hide() {
					visible = false;
					contentView.width = parent.width - projectList.width;
				}

				height: parent.height;
				width: 450
				Layout.minimumWidth: 450
				Rectangle {
					anchors.fill: parent;
					id: rightPaneView
					TabView {
						id: rightPaneTabs
						tabsVisible: true
						antialiasing: true
						anchors.fill: parent
						style: TabViewStyle {
							frameOverlap: 1
							tabBar:
								Rectangle {
									color: "#ededed"
									id: background
								}
							tab: Rectangle {
								color: "#ededed"
								implicitWidth: 80
								implicitHeight: 20
								radius: 2
								Text {
									anchors.centerIn: parent
									text: styleData.title
									color: styleData.selected ? "#7da4cd" : "#202020"
								}
							}
							frame: Rectangle {
							}
						}
					}
				}
			}
		}
	}
}




