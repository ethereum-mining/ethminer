import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0
import QtQuick.Controls.Styles 1.3
import Qt.labs.settings 1.0
import "."

Item {
	property bool renameMode: false;
	property alias sections: sectionRepeater

	ProjectFilesStyle {
		id: projectFilesStyle
	}

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
			color: projectFilesStyle.title.background
			height: projectFilesStyle.title.height
			Layout.fillWidth: true
			Image {
				id: projectIcon
				source: "qrc:/qml/img/dappProjectIcon.png"
				anchors.right: projectTitle.left
				anchors.verticalCenter: parent.verticalCenter
				anchors.rightMargin: 6
				fillMode: Image.PreserveAspectFit
				width: 32
				height: 32
			}

			Text
			{
				id: projectTitle
				color: projectFilesStyle.title.color
				text: projectModel.projectTitle
				anchors.verticalCenter: parent.verticalCenter
				visible: !projectModel.isEmpty;
				anchors.left: parent.left
				anchors.leftMargin: projectFilesStyle.general.leftMargin
				font.family: srcSansProLight.name
				font.pointSize: projectFilesStyle.title.fontSize
				font.weight: Font.Light
			}

			Text
			{
				text: "-"
				anchors.right: parent.right
				anchors.rightMargin: 15
				font.family: srcSansProLight.name
				font.pointSize: projectFilesStyle.title.fontSize
				anchors.verticalCenter: parent.verticalCenter
				font.weight: Font.Light
			}
		}

		Rectangle
		{
			Layout.fillWidth: true
			height: 3
			color: projectFilesStyle.documentsList.background
		}

		Rectangle
		{
			Layout.fillWidth: true
			Layout.fillHeight: true
			color: projectFilesStyle.documentsList.background

			ColumnLayout
			{
				anchors.top: parent.top
				width: parent.width
				spacing: 0
				Repeater {
					model: [qsTr("Contracts"), qsTr("Javascript"), qsTr("Web Pages"), qsTr("Styles"), qsTr("Images"), qsTr("Misc")];
					signal selected(string doc, string groupName)
					signal isCleanChanged(string doc, string groupName, var isClean)
					property int incr: -1;
					id: sectionRepeater
					FilesSection
					{
						id: section;
						sectionName: modelData
						index:
						{
							for (var k in sectionRepeater.model)
							{
								if (sectionRepeater.model[k] === modelData)
									return k;
							}
						}

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
							onContractRenamed: {
								if (modelData === "Contracts")
								{
									var ci = 0;
									for (var si = 0; si < projectModel.listModel.count; si++) {
										var document = projectModel.listModel.get(si);
										if (document.isContract) {
											var compiledDoc = codeModel.contractByDocumentId(document.documentId);
											if (_documentId === document.documentId && _newName !== document.name) {
												document.name = _newName;
												projectModel.listModel.set(si, document);
												sectionModel.set(ci, document);
											}
											ci++;
										}
									}
								}
							}

							onCompilationComplete: {
								if (modelData === "Contracts") {
									var ci = 0;
									for (var si = 0; si < projectModel.listModel.count; si++) {
										var document = projectModel.listModel.get(si);
										if (document.isContract) {
											var compiledDoc = codeModel.contractByDocumentId(document.documentId);
											if (compiledDoc && compiledDoc.documentId === document.documentId && compiledDoc.contract.name !== document.name) {
												document.name = compiledDoc.contract.name;
												projectModel.listModel.set(si, document);
												sectionModel.set(ci, document);
											}
											ci++;
										}
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

							onIsCleanChanged: {
								for (var si = 0; si < sectionModel.count; si++) {
									var document = sectionModel.get(si);
									if (documentId === document.documentId && document.groupName === modelData)
									{
										selManager.isCleanChanged(documentId, modelData, isClean);
										break;
									}
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

