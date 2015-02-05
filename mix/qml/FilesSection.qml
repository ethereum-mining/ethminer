import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0
import QtQuick.Controls.Styles 1.3
import "."


ColumnLayout {
	id: wrapperItem
	signal documentSelected(string doc, string groupName)
	property alias model: filesList.model
	property string sectionName;
	property variant selManager;
	property variant projectModel;
	Layout.fillWidth: true
	Layout.minimumHeight: hiddenHeightTopLevel()
	height: hiddenHeightTopLevel()
	Layout.maximumHeight: hiddenHeightTopLevel()
	spacing: 0

	function hiddenHeightTopLevel()
	{
		return section.state === "hidden" ? Style.documentsList.height : Style.documentsList.height * (model.count + 1);
	}

	function hiddenHeightRepeater()
	{
		return section.state === "hidden" ? 0 : Style.documentsList.height * wrapperItem.model.count;
	}

	function hiddenHeightElement()
	{
		return section.state === "hidden" ? 0 : Style.documentsList.height;
	}

	RowLayout
	{
		anchors.top: parent.top
		id: rowCol
		width: parent.width
		height: Style.documentsList.height
		Text
		{
			id: section
			text: sectionName
			anchors.left: parent.left
			anchors.leftMargin: Style.general.leftMargin
			states: [
				State {
					name: "hidden"
					PropertyChanges { target: filesList; visible: false; }
					PropertyChanges { target: rowCol; Layout.minimumHeight: Style.documentsList.height; Layout.maximumHeight: Style.documentsList.height; height: Style.documentsList.height; }
				}
			]
		}

		MouseArea {
			id: titleMouseArea
			anchors.fill: parent
			hoverEnabled: true
			z: 2
			onClicked: {
				if (section.state === "hidden")
					section.state = "";
				else
					section.state = "hidden";
			}
		}
	}

	ColumnLayout {
		height: wrapperItem.hiddenHeightRepeater()
		Layout.minimumHeight: wrapperItem.hiddenHeightRepeater()
		Layout.preferredHeight: wrapperItem.hiddenHeightRepeater()
		Layout.maximumHeight: wrapperItem.hiddenHeightRepeater()
		width: parent.width
		visible: section.state !== "hidden"
		Repeater
		{
			id: filesList
			visible: section.state !== "hidden"
			Rectangle
			{
				visible: section.state !== "hidden"
				id: rootItem
				Layout.fillWidth: true
				Layout.minimumHeight: wrapperItem.hiddenHeightElement()
				Layout.preferredHeight: wrapperItem.hiddenHeightElement()
				Layout.maximumHeight: wrapperItem.hiddenHeightElement()
				height: wrapperItem.hiddenHeightElement()
				color: isSelected ? Style.documentsList.highlightColor : Style.documentsList.background
				property bool isSelected
				property bool renameMode
				Text {
					id: nameText
					height: parent.height
					visible: !renameMode
					color: rootItem.isSelected ? Style.documentsList.selectedColor : Style.documentsList.color
					text: name;
					font.pointSize: Style.documentsList.fontSize
					anchors.verticalCenter: parent.verticalCenter
					anchors.left: parent.left
					anchors.leftMargin: Style.general.leftMargin
					width: parent.width
					Connections
					{
						target: selManager
						onSelected: {
							if (groupName != sectionName)
								rootItem.isSelected = false;
							else if (doc === documentId)
								rootItem.isSelected = true;
							else
								rootItem.isSelected = false;
						}
					}
				}

				TextInput {
					id: textInput
					text: nameText.text
					visible: renameMode
					anchors.verticalCenter: parent.verticalCenter
					anchors.left: parent.left
					anchors.leftMargin: Style.general.leftMargin
					MouseArea {
						id: textMouseArea
						anchors.fill: parent
						hoverEnabled: true
						z: 2
						onClicked: {
							textInput.forceActiveFocus();
						}
					}

					onVisibleChanged: {
						if (visible) {
							selectAll();
							forceActiveFocus();
						}
					}

					onAccepted: close(true);
					onCursorVisibleChanged: {
						if (!cursorVisible)
							close(false);
					}
					onFocusChanged: {
						if (!focus)
							close(false);
					}
					function close(accept) {
						rootItem.renameMode = false;
						if (accept)
						{
							projectModel.renameDocument(documentId, textInput.text);
							nameText.text = textInput.text;
						}
					}
				}

				MouseArea {
					id: mouseArea
					z: 1
					hoverEnabled: false
					anchors.fill: parent
					acceptedButtons: Qt.LeftButton | Qt.RightButton
					onClicked:{
						if (mouse.button === Qt.RightButton && !isContract)
							contextMenu.popup();
						else if (mouse.button === Qt.LeftButton)
						{
							rootItem.isSelected = true;
							projectModel.openDocument(documentId);
							documentSelected(documentId, groupName);
						}
					}
				}

				Menu {
					id: contextMenu
					MenuItem {
						text: qsTr("Rename")
						onTriggered: {
							rootItem.renameMode = true;
						}
					}
					MenuItem {
						text: qsTr("Delete")
						onTriggered: {
							projectModel.removeDocument(documentId);
						}
					}
				}
			}
		}
	}
}

