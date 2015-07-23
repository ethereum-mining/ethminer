import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0
import QtQuick.Dialogs 1.1
import Qt.labs.settings 1.0
import "js/ProjectModel.js" as ProjectModelCode
import "js/NetworkDeployment.js" as NetworkDeploymentCode

Item {
	id: projectModel

	signal projectClosed
	signal projectLoading(var projectData)
	signal projectLoaded()
	signal documentSaving(var document)
	signal documentChanged(var documentId)
	signal documentOpened(var document)
	signal documentRemoved(var documentId)
	signal documentUpdated(var documentId) //renamed
	signal documentAdded(var documentId)
	signal projectSaving()
	signal projectFileSaving(var projectData)
	signal projectSaved()
	signal projectFileSaved()
	signal newProject(var projectData)
	signal documentSaved(var documentId)
	signal contractSaved(var documentId)
	signal deploymentStarted()
	signal deploymentStepChanged(string message)
	signal deploymentComplete()
	signal deploymentError(string error)
	signal isCleanChanged(var isClean, string documentId)

	property bool isEmpty: (projectPath === "")
	readonly property string projectFileName: ".mix"

	property bool appIsClosing: false
	property bool projectIsClosing: false
	property string projectPath: ""
	property string projectTitle: ""
	property string currentDocumentId: ""
	property var deploymentAddresses: ({})
	property string deploymentDir
	property var listModel: projectListModel
	property var stateListModel: projectStateListModel.model
	property alias stateDialog: projectStateListModel.stateDialog
	property CodeEditorView codeEditor: null
	property var unsavedFiles: []
	property alias newProjectDialog: newProjectDialog
	property int deployedScenarioIndex
	property string applicationUrlEth
	property string applicationUrlHttp
	property string deployBlockNumber
	property var deploymentTrHashes
	property string registerContentHashTrHash
	property string registerUrlTrHash
	property int registerContentHashBlockNumber: -1
	property int registerUrlBlockNumber: -1

	//interface
	function saveAll() { ProjectModelCode.saveAll(); }
	function saveCurrentDocument() { ProjectModelCode.saveCurrentDocument(); }
	function createProject() { ProjectModelCode.createProject(); }
	function closeProject(callBack) { ProjectModelCode.closeProject(callBack); }
	function saveProject() { ProjectModelCode.saveProject(); }
	function saveProjectFile() { ProjectModelCode.saveProjectFile(); }
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
	function getDocumentIdByName(documentName) { return ProjectModelCode.getDocumentIdByName(documentName); }
	function getDocumentIndex(documentId) { return ProjectModelCode.getDocumentIndex(documentId); }
	function addExistingFiles(paths) { ProjectModelCode.doAddExistingFiles(paths); }
	function deployProject() { NetworkDeploymentCode.deployProject(false); }
	function registerToUrlHint(url, gasPrice, callback) { NetworkDeploymentCode.registerToUrlHint(url, gasPrice, callback); }
	function formatAppUrl() { NetworkDeploymentCode.formatAppUrl(url); }

	function cleanDeploymentStatus()
	{
		deployedScenarioIndex = 0
		deployBlockNumber = ""
		deploymentTrHashes = {}
		deploymentAddresses = {}
		deploymentDir = ""
		deploymentDialog.packageStep.packageHash = ""
		deploymentDialog.packageStep.packageBase64 = ""
		deploymentDialog.packageStep.packageDir = ""
		deploymentDialog.packageStep.lastDeployDate = ""
		deploymentDialog.packageStep.localPackageUrl = ""
		saveProject()
		cleanRegisteringStatus()
	}

	function cleanRegisteringStatus()
	{
		applicationUrlEth = ""
		applicationUrlHttp = ""
		registerContentHashTrHash = ""
		registerUrlTrHash = ""
		registerContentHashBlockNumber = -1
		registerUrlBlockNumber = -1
		saveProject()
	}

	Connections {
		target: mainApplication
		onLoaded: {
			if (mainApplication.trackLastProject && projectSettings.lastProjectPath && projectSettings.lastProjectPath !== "")
				projectModel.loadProject(projectSettings.lastProjectPath)
		}
	}

	Connections {
		target: codeEditor
		onIsCleanChanged: {
			for (var i in unsavedFiles)
			{
				if (unsavedFiles[i] === documentId && isClean)
					unsavedFiles.splice(i, 1);
			}
			if (!isClean)
				unsavedFiles.push(documentId);
			isCleanChanged(isClean, documentId);
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

	Connections
	{
		target: fileIo
		property bool saving: false
		onFileChanged:
		{
			fileIo.watchFileChanged(_filePath);
			var documentId = ProjectModelCode.getDocumentByPath(_filePath);
			documentChanged(documentId);
		}
	}

	MessageDialog {
		id: saveMessageDialog
		title: qsTr("Project")
		text: qsTr("Some files require to be saved. Do you want to save changes?");
		standardButtons: StandardButton.Yes | StandardButton.No | StandardButton.Cancel
		icon: StandardIcon.Question
		property var callBack;
		onYes: {
			projectIsClosing = true;
			projectModel.saveAll();
			unsavedFiles = [];
			ProjectModelCode.doCloseProject();
			if (callBack)
				callBack();
		}
		onRejected: {}
		onNo: {
			projectIsClosing = true;
			unsavedFiles = [];
			ProjectModelCode.doCloseProject();
			if (callBack)
				callBack();
		}
	}

	MessageDialog {
		id: deployWarningDialog
		title: qsTr("Project")
		text:
		{
			if (Object.keys(projectModel.deploymentAddresses).length > 0)
				return qsTr("This project has been already deployed to the network. Do you want to redeploy it? (Contract state will be reset)")
			else
				return qsTr("This action will deploy to the network. Do you want to deploy it?")
		}
		icon: StandardIcon.Question
		standardButtons: StandardButton.Ok | StandardButton.Abort
		onAccepted: {
			NetworkDeploymentCode.startDeployProject(true);
		}
	}

	MessageDialog {
		id: deployResourcesDialog
		title: qsTr("Project")
		standardButtons: StandardButton.Ok
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

	Connections
	{
		target: projectModel
		onProjectClosed: {
			projectPath = "";
		}
	}

	Settings {
		id: projectSettings
		property string lastProjectPath;
	}
}
