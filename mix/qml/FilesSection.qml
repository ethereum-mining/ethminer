import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0
import QtQuick.Controls.Styles 1.3
import QtQuick.Dialogs 1.2
import "."


Rectangle
{
	Layout.fillWidth: true
	Layout.minimumHeight: hiddenHeightTopLevel()
	height: hiddenHeightTopLevel()
	Layout.maximumHeight: hiddenHeightTopLevel()
	id: wrapperItem
	signal documentSelected(string doc, string groupName)
	property alias model: filesList.model
	property string sectionName;
	property variant selManager;
	property int index;
	color: index % 2 === 0 ? "transparent" : projectFilesStyle.title.background

	function hiddenHeightTopLevel()
	{
		return section.state === "hidden" ? projectFilesStyle.documentsList.height : projectFilesStyle.documentsList.fileNameHeight * model.count + projectFilesStyle.documentsList.height;
	}

	function hiddenHeightRepeater()
	{
		return section.state === "hidden" ? 0 : projectFilesStyle.documentsList.fileNameHeight * wrapperItem.model.count;
	}

	function hiddenHeightElement()
	{
		return section.state === "hidden" ? 0 : projectFilesStyle.documentsList.fileNameHeight;
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

	ColumnLayout {
		anchors.fill: parent
		spacing: 0

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
			height: projectFilesStyle.documentsList.height
			Layout.fillWidth: true


			Image {
				source: "qrc:/qml/img/opentriangleindicator_filesproject.png"
				width: 15
				sourceSize.width: 12
				id: imgArrow
				anchors.right: section.left
				anchors.rightMargin: 8
				anchors.top: parent.top
				anchors.topMargin: 10
			}

			Text
			{
				id: section
				text: sectionName
				anchors.left: parent.left
				anchors.leftMargin: projectFilesStyle.general.leftMargin
				color: projectFilesStyle.documentsList.sectionColor
				font.family: boldFont.name
				font.pointSize: projectFilesStyle.documentsList.sectionFontSize
				states: [
					State {
						name: "hidden"
						PropertyChanges { target: filesList; visible: false; }
						PropertyChanges { target: rowCol; Layout.minimumHeight: projectFilesStyle.documentsList.height; Layout.maximumHeight: projectFilesStyle.documentsList.height; height: projectFilesStyle.documentsList.height; }
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
					color: isSelected ? projectFilesStyle.documentsList.highlightColor : "transparent"
					property bool isSelected
					property bool renameMode

					Row {
						spacing: 3
						anchors.verticalCenter: parent.verticalCenter
						anchors.fill: parent
						anchors.left: parent.left
						anchors.leftMargin: projectFilesStyle.general.leftMargin + 2
						Text {
							id: nameText
							height: parent.height
							visible: !renameMode
							color: rootItem.isSelected ? projectFilesStyle.documentsList.selectedColor : projectFilesStyle.documentsList.color
							text: name;
							font.family: fileNameFont.name
							font.pointSize: projectFilesStyle.documentsList.fontSize
							verticalAlignment:  Text.AlignVCenter

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
								onIsCleanChanged: {
									if (groupName === sectionName && doc === documentId)
										editStatusLabel.visible = !isClean;
								}
							}
						}

						DefaultLabel {
							id: editStatusLabel
							visible: false
							color: rootItem.isSelected ? projectFilesStyle.documentsList.selectedColor : projectFilesStyle.documentsList.color
							verticalAlignment:  Text.AlignVCenter
							text: "*"
							width: 10
							height: parent.height
						}
					}

					TextInput {
						id: textInput
						text: nameText.text
						visible: renameMode
						anchors.verticalCenter: parent.verticalCenter
						anchors.left: parent.left
						anchors.leftMargin: projectFilesStyle.general.leftMargin
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
							if (mouse.button === Qt.RightButton)
							{
								if (isContract)
									contextMenuContract.popup();
								else
									contextMenu.popup();
							}
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
								deleteConfirmation.open();
							}
						}
					}

					Menu {
						id: contextMenuContract
						MenuItem {
							text: qsTr("Delete")
							onTriggered: {
								deleteConfirmation.open();
							}
						}
					}

					MessageDialog
					{
						id: deleteConfirmation
						text: qsTr("Are you sure to delete this file ?")
						standardButtons: StandardIcon.Ok | StandardIcon.Cancel
						onAccepted:
						{
							projectModel.removeDocument(documentId);
							wrapperItem.removeDocument(documentId);
						}
					}
				}
			}
		}
	}
}
