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
	Layout.fillWidth: true
	Layout.minimumHeight: hiddenHeightTopLevel()
	height: hiddenHeightTopLevel()
	Layout.maximumHeight: hiddenHeightTopLevel()
	spacing: 0

	function hiddenHeightTopLevel()
	{
		return section.state === "hidden" ? ProjectFilesStyle.documentsList.height : ProjectFilesStyle.documentsList.fileNameHeight * model.count + ProjectFilesStyle.documentsList.height;
	}

	function hiddenHeightRepeater()
	{
		return section.state === "hidden" ? 0 : ProjectFilesStyle.documentsList.fileNameHeight * wrapperItem.model.count;
	}

	function hiddenHeightElement()
	{
		return section.state === "hidden" ? 0 : ProjectFilesStyle.documentsList.fileNameHeight;
	}

	function getDocumentIndex(documentId)
	{
		for (var i = 0; i < model.count; i++)
			if (model.get(i).documentId === documentId)
				return i;
		return -1;
	}

	function removeDocument(documentId)
	{
		var i = getDocumentIndex(documentId);
		if (i !== -1)
			model.remove(i);
	}

	SourceSansProRegular
	{
		id: fileNameFont
	}

	SourceSansProBold
	{
		id: boldFont
	}

	RowLayout
	{
		anchors.top: parent.top
		id: rowCol
		width: parent.width
		height: ProjectFilesStyle.documentsList.height

		Image {
			source: "qrc:/qml/img/opentriangleindicator_filesproject.png"
			width: 15
			sourceSize.width: 12
			id: imgArrow
			anchors.right: section.left
			anchors.rightMargin: 8
			anchors.top: parent.top
			anchors.topMargin: 6
		}

		Text
		{
			id: section
			text: sectionName
			anchors.left: parent.left
			anchors.leftMargin: ProjectFilesStyle.general.leftMargin
			color: ProjectFilesStyle.documentsList.sectionColor
			font.family: boldFont.name
			font.pointSize: ProjectFilesStyle.documentsList.sectionFontSize
			states: [
				State {
					name: "hidden"
					PropertyChanges { target: filesList; visible: false; }
					PropertyChanges { target: rowCol; Layout.minimumHeight: ProjectFilesStyle.documentsList.height; Layout.maximumHeight: ProjectFilesStyle.documentsList.height; height: ProjectFilesStyle.documentsList.height; }
					PropertyChanges { target: imgArrow; source: "qrc:/qml/img/closedtriangleindicator_filesproject.png" }
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
		spacing: 0
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
				color: isSelected ? ProjectFilesStyle.documentsList.highlightColor : ProjectFilesStyle.documentsList.background
				property bool isSelected
				property bool renameMode
				Text {
					id: nameText
					height: parent.height
					visible: !renameMode
					color: rootItem.isSelected ? ProjectFilesStyle.documentsList.selectedColor : ProjectFilesStyle.documentsList.color
					text: name;
					font.family: fileNameFont.name
					font.pointSize: ProjectFilesStyle.documentsList.fontSize
					anchors.verticalCenter: parent.verticalCenter
					verticalAlignment:  Text.AlignVCenter
					anchors.left: parent.left
					anchors.leftMargin: ProjectFilesStyle.general.leftMargin + 2
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

							if (rootItem.isSelected && section.state === "hidden")
								section.state = "";
						}
					}
				}

				TextInput {
					id: textInput
					text: nameText.text
					visible: renameMode
					anchors.verticalCenter: parent.verticalCenter
					anchors.left: parent.left
					anchors.leftMargin: ProjectFilesStyle.general.leftMargin
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
							var i = getDocumentIndex(documentId);
							projectModel.renameDocument(documentId, textInput.text);
							wrapperItem.model.set(i, projectModel.getDocument(documentId));
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
							wrapperItem.removeDocument(documentId);
						}
					}
				}
			}
		}
	}
}

