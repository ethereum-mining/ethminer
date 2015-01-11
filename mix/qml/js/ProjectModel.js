/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file ProjectModel.js
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */

function saveAll() {
	saveProject();
}

function createProject() {
	newProjectDialog.open();
}

function browseProject() {
	openProjectFileDialog.open();
}

function closeProject() {
	if (!isEmpty) {
		if (haveUnsavedChanges)
			saveMessageDialog.open();
		else
			doCloseProject();
	}
}

function saveProject() {
	if (!isEmpty) {
		projectSaving(projectData);
		var json = JSON.stringify(projectData);
		var projectFile = projectPath + projectFileName;
		fileIo.writeFile(projectFile, json);
	}
}

function loadProject(path) {
	closeProject();
	console.log("loading project at " + path);
	var projectFile = path + projectFileName;
	var json = fileIo.readFile(projectFile);
	projectData = JSON.parse(json);
	if (!projectData.title) {
		var parts = path.split("/");
		projectData.title = parts[parts.length - 2];
	}
	projectPath = path;
	if (!projectData.files)
		projectData.files = [];

	for(var i = 0; i < projectData.files.length; i++) {
		addFile(projectData.files[i]);
	}
	projectSettings.lastProjectPath = path;
	projectLoaded();
}

function addExistingFile() {
	addExistingFileDialog().open();
}

function addProjectFiles(files) {
	for(var i = 0; i < files.length; i++)
		addFile(files[i]);
}

function addFile(fileName) {
	var p = projectPath + fileName;
	var fileData = {
		contract: false,
		path: p,
		name: fileName,
		documentId: fileName,
		isText: true,
		isContract: fileName.substring(fileName.length - 4, fileName.length) === ".sol",
	};

	projectListModel.append(fileData);
}

function openDocument(documentId) {
	for (var i = 0; i < projectListModel.count; i++)
		if (projectListModel.get(i).documentId === documentId)
			documentOpened(projectListModel.get(i));
}

function doCloseProject() {
	console.log("closing project");
	projectListModel.clear();
	projectPath = "";
	projectData = null;
	projectClosed();
}

function doCreateProject(title, path) {
	closeProject();
	console.log("creating project " + title + " at " + path);
	if (path[path.length - 1] !== "/")
		path += "/";
	var dirPath = path + title + "/";
	fileIo.makeDir(dirPath);
	var projectFile = dirPath + projectFileName;

	var indexFile = dirPath + "index.html";
	var contractsFile = dirPath + "contracts.sol";
	var projectData = {
		title: title,
		files: [ "contracts.sol", "index.html" ]
	};

	fileIo.writeFile(indexFile, "<html></html>");
	fileIo.writeFile(contractsFile, "contract MyContract {}");
	var json = JSON.stringify(projectData);
	fileIo.writeFile(projectFile, json);
	loadProject(dirPath);
}

function doAddExistingFiles(files) {
	for(var i = 0; i < files.length; i++) {
		var sourcePath = files[i];
		var sourceFileName = sourcePath.substring(sourcePath.lastIndexOf("/") + 1, sourcePath.length);
		var destPath = projectPath + sourceFileName;
		fileIo.copyFile(sourcePath, destPath);
		addFile(sourceFileName);
	}
}
