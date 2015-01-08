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

	property bool isEmpty: projectFile === ""
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

	function closeProject() {
		console.log("closing project");
		if (haveUnsavedChanges)
			saveMessageDialog.open();
		else
			doCloseProject();
	}

	function saveProject() {
		if (!isEmpty) {
			var json = JSON.stringify(projectData);
			fileIo.writeFile(projectFile, json)
		}
	}

	function loadProject(path) {
		if (!isEmpty)
			closeProject();
		console.log("loading project at " + path);
		var json = fileIo.readFile(path);
		projectData = JSON.parse(json);
		projectFile = path;
		if (!projectData.files)
			projectData.files = [];

		for(var i = 0; i < projectData.files; i++) {
			var p = projectData.files[i];
			projectListModel.append({
				path: p,
				name: p.substring(p.lastIndexOf("/") + 1, p.length)
			});
		}
		onProjectLoaded();
	}

	function doCloseProject() {
		projectListModel.clear();
		projectFile = "";
		projectData = null;
		projectClosed();
	}

	function doCreateProject(title, path) {
		if (!isEmpty)
			closeProject();
		console.log("creating project " + title + " at " + path);
		if (path[path.length - 1] !== "/")
			path += "/";
		var dirPath = path + title;
		fileIo.makeDir(dirPath);
		var projectFile = dirPath + "/" + projectFileName;
		fileIo.writeFile(projectFile, "");
		loadProject(projectFile);
	}

	NewProjectDialog {
		id: newProjectDialog
		visible: false
		onAccepted: {
			var title = newProjectDialog.projectTitle;
			var path = newProjectDialog.projectPath;
			projectModel.doCreateProject(title, path);
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

	Component {
		id: renderDelegate
		Item {
			id: wrapperItem
			height: 20
			width: parent.width
			RowLayout {
				anchors.fill: parent
				Text {
					Layout.fillWidth: true
					Layout.fillHeight: true
					text: title
					font.pointSize: 12
					verticalAlignment: Text.AlignBottom
				}
			}
		}
	}

	Settings {
		id: projectSettings
		property string lastProjectPath;
	}
}
