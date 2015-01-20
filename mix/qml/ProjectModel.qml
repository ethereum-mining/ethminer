pragma Singleton

import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0
import QtQuick.Dialogs 1.1
import Qt.labs.settings 1.0

import "js/ProjectModel.js" as ProjectModelCode

Item {
	id: projectModel

	signal projectClosed
	signal projectLoaded(var projectData)
	signal documentOpened(var document)
	signal documentRemoved(var documentId)
	signal documentUpdated(var documentId) //renamed
	signal documentAdded(var documentId)
	signal projectSaving(var projectData)
	signal projectSaved()
	signal documentSaved(var documentId)

	property bool isEmpty: (projectPath === "")
	readonly property string projectFileName: ".mix"

	property bool haveUnsavedChanges: false
	property string projectPath: ""
	property string projectTitle: ""
	property var listModel: projectListModel

	//interface
	function saveAll() { ProjectModelCode.saveAll(); }
	function createProject() { ProjectModelCode.createProject(); }
	function browseProject() { ProjectModelCode.browseProject(); }
	function closeProject() { ProjectModelCode.closeProject(); }
	function saveProject() { ProjectModelCode.saveProject(); }
	function loadProject(path) { ProjectModelCode.loadProject(path); }
	function addExistingFile() { ProjectModelCode.addExistingFile(); }
	function newHtmlFile() { ProjectModelCode.newHtmlFile(); }
	function newJsFile() { ProjectModelCode.newJsFile(); }
	//function newContract() { ProjectModelCode.newContract(); }
	function openDocument(documentId) { ProjectModelCode.openDocument(documentId); }
	function renameDocument(documentId, newName) { ProjectModelCode.renameDocument(documentId, newName); }
	function removeDocument(documentId) { ProjectModelCode.removeDocument(documentId); }
	function getDocument(documentId) { return ProjectModelCode.getDocument(documentId); }

	Connections {
		target: appContext
		onAppLoaded: {
			if (projectSettings.lastProjectPath)
				loadProject(projectSettings.lastProjectPath)
		}
	}

	NewProjectDialog {
		id: newProjectDialog
		visible: false
		onAccepted: {
			var title = newProjectDialog.projectTitle;
			var path = newProjectDialog.projectPath;
			ProjectModelCode.doCreateProject(title, path);
		}
	}

	MessageDialog {
		id: saveMessageDialog
		title: qsTr("Project")
		text: qsTr("Do you want to save changes?")
		standardButtons: StandardButton.Ok | StandardButton.Cancel
		icon: StandardIcon.Question
		onAccepted: {
			projectModel.saveAll();
			ProjectModelCode.doCloseProject();
		}
		onRejected: {
			ProjectModelCode.doCloseProject();
		}
	}

	ListModel {
		id: projectListModel
	}

	Settings {
		id: projectSettings
		property string lastProjectPath;
	}

	FileDialog {
		id: openProjectFileDialog
		visible: false
		title: qsTr("Open a project")
		selectFolder: true
		onAccepted: {
			var path = openProjectFileDialog.fileUrl.toString();
			path += "/";
			loadProject(path);
		}
	}

	FileDialog {
		id: addExistingFileDialog
		visible: false
		title: qsTr("Add a file")
		selectFolder: false
		onAccepted: {
			var paths = addExistingFileDialog.fileUrls;
			ProjectModelCode.doAddExistingFiles(paths);
		}
	}
}
