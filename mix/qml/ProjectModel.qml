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
	signal projectLoading(var projectData)
	signal projectLoaded()
	signal documentOpened(var document)
	signal documentRemoved(var documentId)
	signal documentUpdated(var documentId) //renamed
	signal documentAdded(var documentId)
	signal projectSaving(var projectData)
	signal projectSaved()
	signal newProject(var projectData)
	signal documentSaved(var documentId)
	signal deploymentStarted()
	signal deploymentStepChanged(string message)
	signal deploymentComplete()
	signal deploymentError(string error)

	property bool isEmpty: (projectPath === "")
	readonly property string projectFileName: ".mix"

	property bool haveUnsavedChanges: false
	property string projectPath: ""
	property string projectTitle: ""
	property string currentDocumentId: ""
	property var deploymentAddresses: []
	property var listModel: projectListModel
	property var stateListModel: projectStateListModel.model
	property CodeEditorView codeEditor: null

	//interface
	function saveAll() { ProjectModelCode.saveAll(); }
	function createProject() { ProjectModelCode.createProject(); }
	function closeProject() { ProjectModelCode.closeProject(); }
	function saveProject() { ProjectModelCode.saveProject(); }
	function loadProject(path) { ProjectModelCode.loadProject(path); }
	function newHtmlFile() { ProjectModelCode.newHtmlFile(); }
	function newJsFile() { ProjectModelCode.newJsFile(); }
	function newCssFile() { ProjectModelCode.newCssFile(); }
	function newContract() { ProjectModelCode.newContract(); }
	function openDocument(documentId) { ProjectModelCode.openDocument(documentId); }
	function openNextDocument() { ProjectModelCode.openNextDocument(); }
	function openPrevDocument() { ProjectModelCode.openPrevDocument(); }
	function renameDocument(documentId, newName) { ProjectModelCode.renameDocument(documentId, newName); }
	function removeDocument(documentId) { ProjectModelCode.removeDocument(documentId); }
	function getDocument(documentId) { return ProjectModelCode.getDocument(documentId); }
	function getDocumentIndex(documentId) { return ProjectModelCode.getDocumentIndex(documentId); }
	function addExistingFiles(paths) { ProjectModelCode.doAddExistingFiles(paths); }
	function deployProject() { ProjectModelCode.deployProject(false); }
	function registerToUrlHint() { ProjectModelCode.registerToUrlHint(); }

	Connections {
		target: appContext
		onAppLoaded: {
			if (projectSettings.lastProjectPath)
				projectModel.loadProject(projectSettings.lastProjectPath)
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

	MessageDialog {
		id: deployWarningDialog
		property bool redeploy
		title: qsTr("Project")
		text:
		{
			if (Object.keys(projectModel.deploymentAddresses).length > 0)
			{
				redeploy = true
				standardButtons = StandardButton.Ok | StandardButton.Reset | StandardButton.Abort;
				return qsTr("This project has been already deployed to the network. Do you want to repackage the resources only, or also reset the deployed contract to its initial state?")
			}
			else
			{
				redeploy = false;
				standardButtons = StandardButton.Ok | StandardButton.Abort;
				return qsTr("This action will deploy to the network. Do you want to deploy it?")
			}
		}
		icon: StandardIcon.Question
		onAccepted: {
			ProjectModelCode.startDeployProject(!redeploy);
		}
		onReset: {
			ProjectModelCode.startDeployProject(true);
		}
	}

	MessageDialog {
		id: deployRessourcesDialog
		title: qsTr("Project")
		standardButtons: StandardButton.Ok
		icon: StandardIcon.Info
	}

	DeploymentDialog
	{
		id: deploymentDialog
	}

	ListModel {
		id: projectListModel
	}

	StateListModel {
		id: projectStateListModel
	}

	Settings {
		id: projectSettings
		property string lastProjectPath;
	}
}
