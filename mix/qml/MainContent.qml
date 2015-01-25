import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1
import CodeEditorExtensionManager 1.0
import Qt.labs.settings 1.0

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

	property alias rightViewVisible : rightView.visible
	property alias webViewVisible : webPreview.visible
	property bool webViewHorizontal : codeWebSplitter.orientation === Qt.Vertical //vertical splitter positions elements vertically, splits screen horizontally

	onWidthChanged:
	{
		if (rightView.visible)
			contentView.width = parent.width - projectList.width - rightView.width;
		else
			contentView.width = parent.width - projectList.width;
	}

	function toggleRightView() {
		if (!rightView.visible)
			rightView.show();
		else
			rightView.hide();
	}

	function ensureRightView() {
		if (!rightView.visible)
			rightView.show();
	}

	function hideRightView() {
		if (rightView.visible)
			rightView.hide();
	}

	function toggleWebPreview() {
		webPreview.visible = !webPreview.visible;
	}

	function toggleWebPreviewOrientation() {
		codeWebSplitter.orientation = (codeWebSplitter.orientation === Qt.Vertical ? Qt.Horizontal : Qt.Vertical);
	}

	function rightViewVisible()	{
		return rightView.visible;
	}

	CodeEditorExtensionManager {
		headerView: headerPaneTabs;
		rightView: rightPaneTabs;
	}

	Settings {
		id: mainLayoutSettings
		property alias codeWebOrientation: codeWebSplitter.orientation
		property alias webWidth: webPreview.width
		property alias webHeight: webPreview.height
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

		Rectangle {
			Layout.fillWidth: true
			Layout.preferredHeight: root.height - headerView.height;

			Settings {
				id: splitSettings
				property alias projectWidth: projectList.width
				property alias contentViewWidth: contentView.width
				property alias rightViewWidth: rightView.width
			}

			ProjectList	{
				anchors.left: parent.left
				id: projectList
				width: 200
				height: parent.height
				Layout.minimumWidth: 200
			}

			Splitter
			{
				id: resizeLeft
				itemToStick: projectList
				itemMinimumWidth: projectList.Layout.minimumWidth
				direction: "right"
				brother: contentView
				color: "#a2a2a2"
			}

			Rectangle {
				anchors.left: projectList.right
				id: contentView
				width: parent.width - projectList.width
				height: parent.height
				SplitView {
					 handleDelegate: Rectangle {
						width: 4
						height: 4
						color: "#cccccc"
					 }
					id: codeWebSplitter
					anchors.fill: parent
					orientation: Qt.Vertical
					CodeEditorView {
						height: parent.height * 0.6
						anchors.top: parent.top
						Layout.fillWidth: true
						Layout.fillHeight: true
					}
					WebPreview {
						id: webPreview
						height: parent.height * 0.4
						Layout.fillWidth: codeWebSplitter.orientation === Qt.Vertical
						Layout.fillHeight: codeWebSplitter.orientation === Qt.Horizontal
						Layout.minimumHeight: 200
						Layout.minimumWidth: 200
					}
				}
			}

			Splitter
			{
				id: resizeRight
				visible: false;
				itemToStick: rightView
				itemMinimumWidth: rightView.Layout.minimumWidth
				direction: "left"
				brother: contentView
				color: "#a2a2a2"
			}

			Rectangle {
				visible: false;
				id: rightView;

				Keys.onEscapePressed: hide()

				function show() {
					visible = true;
					resizeRight.visible = true;
					contentView.width = parent.width - projectList.width - rightView.width;
				}

				function hide() {
					resizeRight.visible = false;
					visible = false;
					contentView.width = parent.width - projectList.width;
				}

				height: parent.height;
				width: 515
				Layout.minimumWidth: 515
				anchors.right: parent.right
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




