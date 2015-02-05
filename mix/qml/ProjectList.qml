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

		Item {
			id: selManager
			signal selected(string doc, string groupName)
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
				FilesSection
				{
					sectionName: "Contracts"
					projectModel: projectModelConnection
					model: ctrModel
					selManager: selManager
					onDocumentSelected: {
						selManager.selected(doc, groupName);
					}
				}

				ListModel
				{
					id: ctrModel
				}

				FilesSection
				{
					id: javascriptSection
					sectionName: "Javascript"
					projectModel: projectModelConnection
					model: javascriptModel
					selManager: selManager
					onDocumentSelected: {
						selManager.selected(doc, groupName);
					}
				}

				ListModel
				{
					id: javascriptModel
				}

				FilesSection
				{
					id: htmlSection
					sectionName: "HTML"
					projectModel: projectModelConnection
					model: htmlModel
					selManager: selManager
					onDocumentSelected: {
						selManager.selected(doc, groupName);
					}
				}

				ListModel
				{
					id: htmlModel
				}

				FilesSection
				{
					id: stylesSection
					sectionName: "Styles"
					model: styleModel
					projectModel: projectModelConnection
					selManager: selManager
					onDocumentSelected: {
						selManager.selected(doc, groupName);
					}
				}

				ListModel
				{
					id: styleModel
				}

				FilesSection
				{
					id: imgSection
					projectModel: projectModelConnection
					sectionName: "Images"
					model: imgModel
					selManager: selManager
					onDocumentSelected: {
						selManager.selected(doc, groupName);
					}
				}

				ListModel
				{
					id: imgModel
				}
			}
		}
	}

	Connections {
		id: projectModelConnection
		signal projectFilesLoaded;
		target: projectModel
		function addDocToSubModel(index)
		{
			var item = projectModel.listModel.get(index);
			if (item.groupName === "Contracts")
				ctrModel.append(item);
			else if (item.groupName === "HTML")
				htmlModel.append(item)
			else if (item.groupName === "Javascript")
				javascriptModel.append(item)
			else if (item.groupName === "Images")
				imagesModel.append(item)
			else if (item.groupName === "Styles")
				stylesModel.append(item)
		}

		onProjectLoaded: {
			projectModel.openDocument(projectModel.listModel.get(0).documentId);
			ctrModel.clear();
			htmlModel.clear();
			javascriptModel.clear();
			imgModel.clear();
			styleModel.clear();
			for (var k = 0; k < projectModel.listModel.count; k++)
				addDocToSubModel(k);
			projectFilesLoaded();
		}
		onDocumentAdded:
		{
			addDocToSubModel(projectModel.getDocumentIndex(documentId));
		}
	}
}

