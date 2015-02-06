import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0
import QtQuick.Controls.Styles 1.3
import Qt.labs.settings 1.0
import "."

Item {
	property bool renameMode: false;
	ColumnLayout {
		anchors.fill: parent
		id: filesCol
		spacing: 0
		Rectangle
		{
			color: Style.title.background
			height: Style.title.height
			Layout.fillWidth: true
			Text
			{
				color: Style.title.color
				text: projectModel.projectTitle
				anchors.verticalCenter: parent.verticalCenter
				visible: !projectModel.isEmpty;
				anchors.left: parent.left
				anchors.leftMargin: Style.general.leftMargin
			}
		}

		Rectangle
		{
			Layout.fillWidth: true
			height: 10
			color: Style.documentsList.background
		}



		Rectangle
		{
			Layout.fillWidth: true
			Layout.fillHeight: true
			color: Style.documentsList.background

			ColumnLayout
			{
				anchors.top: parent.top
				width: parent.width
				spacing: 0

				Repeater {
					model: ["Contracts", "Javascript", "HTML", "Styles", "Images"]
					signal selected(string doc, string groupName)
					id: sectionRepeater
					FilesSection
					{
						sectionName: modelData
						model: sectionModel
						selManager: sectionRepeater

						onDocumentSelected: {
							selManager.selected(doc, groupName);
						}

						ListModel
						{
							id: sectionModel
						}

						Connections {
							target: codeModel
							onContractNameChanged: {
								if (modelData === "Contracts")
								{
									var ctr = projectModel.listModel.get(0);
									ctr.name = _newName;
									projectModel.listModel.set(0, ctr);
									sectionModel.set(0, ctr);
								}
							}
						}

						Connections {
							id: projectModelConnection
							target: projectModel

							function addDocToSubModel()
							{
								for (var k = 0; k < projectModel.listModel.count; k++)
								{
									var item = projectModel.listModel.get(k);
									if (item.groupName === modelData)
										sectionModel.append(item);
								}
							}

							onProjectLoaded: {
								addDocToSubModel();
								if (modelData === "Contracts")
								{
									var selItem = projectModel.listModel.get(0);
									projectModel.openDocument(selItem.documentId);
									sectionRepeater.selected(selItem.documentId, modelData);
								}
							}

							onDocumentAdded:
							{
								var newDoc = projectModel.getDocumentIndex(documentId);
								if (newDoc.groupName === modelData)
									ctrModel.append(newDoc);
							}
						}
					}
				}
			}
		}
	}
}

