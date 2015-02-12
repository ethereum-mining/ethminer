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

		SourceSansProLight
		{
			id: srcSansProLight
		}

		Rectangle
		{
			color: ProjectFilesStyle.title.background
			height: ProjectFilesStyle.title.height
			Layout.fillWidth: true
			Image {
				id: projectIcon
				source: "qrc:/qml/img/projecticon.png"
				sourceSize.height: 30
				anchors.right: projectTitle.left
				anchors.verticalCenter: parent.verticalCenter
				anchors.rightMargin: 6
			}

			Text
			{
				id: projectTitle
				color: ProjectFilesStyle.title.color
				text: projectModel.projectTitle
				anchors.verticalCenter: parent.verticalCenter
				visible: !projectModel.isEmpty;
				anchors.left: parent.left
				anchors.leftMargin: ProjectFilesStyle.general.leftMargin
				font.family: srcSansProLight.name
				font.pointSize: ProjectFilesStyle.title.fontSize
				font.weight: Font.Light
			}

			Text
			{
				text: "-"
				anchors.right: parent.right
				anchors.rightMargin: 15
				font.family: srcSansProLight.name
				font.pointSize: ProjectFilesStyle.title.fontSize
				anchors.verticalCenter: parent.verticalCenter
				font.weight: Font.Light
			}
		}

		Rectangle
		{
			Layout.fillWidth: true
			height: 10
			color: ProjectFilesStyle.documentsList.background
		}



		Rectangle
		{
			Layout.fillWidth: true
			Layout.fillHeight: true
			color: ProjectFilesStyle.documentsList.background

			ColumnLayout
			{
				anchors.top: parent.top
				width: parent.width
				spacing: 0

				Repeater {
					model: [qsTr("Contracts"), qsTr("Javascript"), qsTr("Web Pages"), qsTr("Styles"), qsTr("Images"), qsTr("Misc")];
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
							onCompilationComplete: {
								if (modelData === "Contracts")
								{
									var ctr = projectModel.listModel.get(0);
									if (codeModel.code.contract.name !== ctr.name)
									{
										ctr.name = codeModel.code.contract.name;
										projectModel.listModel.set(0, ctr);
										sectionModel.set(0, ctr);
									}
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

							onDocumentOpened: {
								if (document.groupName === modelData)
									sectionRepeater.selected(document.documentId, modelData);
							}

							onNewProject: {
								sectionModel.clear();
							}

							onProjectClosed: {
								sectionModel.clear();
							}

							onProjectLoaded: {
								sectionModel.clear();
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
								var newDoc = projectModel.getDocument(documentId);
								if (newDoc.groupName === modelData)
								{
									sectionModel.append(newDoc);
									projectModel.openDocument(newDoc.documentId);
									sectionRepeater.selected(newDoc.documentId, modelData);
								}

							}
						}
					}
				}
			}
		}
	}
}

