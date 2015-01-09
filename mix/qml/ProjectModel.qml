pragma Singleton

import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0
import QtQuick.Dialogs 1.1
import Qt.labs.settings 1.0

Item {
	id: projectModel

	signal projectClosed
	signal projectLoaded
	signal documentOpen(var document)

	property bool isEmpty: (projectFile === "")
	readonly property string projectFileName: ".mix"

	property bool haveUnsavedChanges: false
	property string projectFile: ""
	property var projectData: null
	property var listModel: projectListModel

	function saveAll() {
		saveProject();
	}

	function createProject() {
		closeProject();
		newProjectDialog.open();
	}

	function browseProject() {
		openProjectFileDialog.open();
	}

	function closeProject() {
		if (!isEmpty) {
			console.log("closing project");
			if (haveUnsavedChanges)
				saveMessageDialog.open();
			else
				doCloseProject();
		}
	}

	function saveProject() {
		if (!isEmpty) {
			var json = JSON.stringify(projectData);
			fileIo.writeFile(projectFile, json);
		}
	}

	function loadProject(path) {
		closeProject();
		console.log("loading project at " + path);
		var json = fileIo.readFile(path);
		projectData = JSON.parse(json);
		projectFile = path;
		if (!projectData.files)
			projectData.files = [];

		for(var i = 0; i < projectData.files.length; i++) {
			var p = projectData.files[i];
			addFile(p);
		}
		projectSettings.lastProjectPath = projectFile;
		projectLoaded();
	}

	function addExistingFile() {
		addExistingFileDialog().open();
	}

	function addProjectFiles(files) {
		for(var i = 0; i < files.length; i++)
			addFile(files[i]);
	}

	function addFile(file) {
		var p = file;
		var fileData = {
			contract: false,
			path: p,
			name: p.substring(p.lastIndexOf("/") + 1, p.length),
			documentId: p,
			isText: true,
			isContract: p.substring(p.length - 4, p.length) === ".sol",
		};

		projectListModel.append(fileData);
	}

	function doCloseProject() {
		projectListModel.clear();
		projectFile = "";
		projectData = null;
		projectClosed();
	}

	function doCreateProject(title, path) {
		closeProject();
		console.log("creating project " + title + " at " + path);
		if (path[path.length - 1] !== "/")
			path += "/";
		var dirPath = path + title;
		fileIo.makeDir(dirPath);
		var projectFile = dirPath + "/" + projectFileName;

		var indexFile = dirPath + "/index.html";
		var contractsFile = dirPath + "/contracts.sol";
		var projectData = {
			files: [ indexFile, contractsFile ]
		};

		fileIo.writeFile(indexFile, "<html></html>");
		fileIo.writeFile(contractsFile, "contract MyContract {}");
		var json = JSON.stringify(projectData);
		fileIo.writeFile(projectFile, json);
		loadProject(projectFile);
	}

	Component.onCompleted: {
		if (projectSettings.lastProjectPath)
			loadProject(projectSettings.lastProjectPath)
	}

	NewProjectDialog {
		id: newProjectDialog
		visible: false
		onAccepted: {
			var title = newProjectDialog.projectTitle;
			var path = newProjectDialog.projectPath;
			doCreateProject(title, path);
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
			projectModel.doCloseProject();
		}
		onRejected: {
			projectModel.doCloseProject();
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
			path += "/" + projectFileName;
				loadProject(path);
		}
	}

	FileDialog {
		id: addExistingFileDialog
		visible: false
		title: qsTr("Add a file")
		selectFolder: false
		onAccepted: {
			var paths = openProjectFileDialog.fileUrls;
				addProjectFiles(paths);
		}
	}


}
