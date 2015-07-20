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

var htmlTemplate = "<html>\n<head>\n<script>\n</script>\n</head>\n<body>\n<script>\n</script>\n</body>\n</html>";
var contractTemplate = "contract Contract {\n}\n";

function saveCurrentDocument()
{
	var doc = projectListModel.get(getDocumentIndex(currentDocumentId));
	documentSaving(doc);
	if (doc.isContract)
		contractSaved(currentDocumentId);
	else
		documentSaved(currentDocumentId);
}

function saveAll() {
	saveProject();
}

function createProject() {
	newProjectDialog.open();
}

function closeProject(callBack) {
	if (!isEmpty && unsavedFiles.length > 0)
	{
		saveMessageDialog.callBack = callBack;
		saveMessageDialog.open();
	}
	else
	{
		projectIsClosing = true;
		doCloseProject();
		if (callBack)
			callBack();
	}
}

function saveProject() {
	if (!isEmpty) {
		projectSaving();
		var projectData = saveProjectFile();
		if (projectData !== null)
		{
			projectSaved();
		}
	}
}

function saveProjectFile()
{
	if (!isEmpty) {
		var projectData = {
			files: [],
			title: projectTitle,
			deploymentAddresses: deploymentAddresses,
			applicationUrlEth: projectModel.applicationUrlEth,
			applicationUrlHttp: projectModel.applicationUrlHttp,
			packageHash: deploymentDialog.packageStep.packageHash,
			packageBase64: deploymentDialog.packageStep.packageBase64,
			deploymentDir: deploymentDialog.packageStep.packageDir,
			lastPackageDate:  deploymentDialog.packageStep.lastDeployDate,
			deployBlockNumber: projectModel.deployBlockNumber,
			localPackageUrl: deploymentDialog.packageStep.localPackageUrl,
			deploymentTrHashes: JSON.stringify(projectModel.deploymentTrHashes),
			registerContentHashTrHash: projectModel.registerContentHashTrHash,
			registerUrlTrHash: projectModel.registerUrlTrHash,
			registerContentHashBlockNumber: projectModel.registerContentHashBlockNumber,
			registerUrlBlockNumber: projectModel.registerUrlBlockNumber
		};
		for (var i = 0; i < projectListModel.count; i++)
			projectData.files.push({
									   title: projectListModel.get(i).name,
									   fileName: projectListModel.get(i).fileName,
								   });

		projectFileSaving(projectData);
		var json = JSON.stringify(projectData, null, "\t");
		var projectFile = projectPath + projectFileName;
		fileIo.writeFile(projectFile, json);
		projectFileSaved(projectData);
		return projectData;
	}
	return null;
}

function loadProject(path) {
	closeProject(function() {
		console.log("Loading project at " + path);
		var projectFile = path + projectFileName;
		var json = fileIo.readFile(projectFile);
		if (!json)
			return;
		var projectData = JSON.parse(json);
		if (projectData.deploymentDir)
			projectModel.deploymentDir = projectData.deploymentDir
		if (projectData.packageHash)
			deploymentDialog.packageStep.packageHash =  projectData.packageHash
		if (projectData.packageBase64)
			deploymentDialog.packageStep.packageBase64 =  projectData.packageBase64
		if (projectData.applicationUrlEth)
			projectModel.applicationUrlEth = projectData.applicationUrlEth
		if (projectData.applicationUrlHttp)
			projectModel.applicationUrlHttp = projectData.applicationUrlHttp
		if (projectData.lastPackageDate)
			deploymentDialog.packageStep.lastDeployDate = projectData.lastPackageDate
		if (projectData.deployBlockNumber)
			projectModel.deployBlockNumber = projectData.deployBlockNumber
		if (projectData.localPackageUrl)
			deploymentDialog.packageStep.localPackageUrl =  projectData.localPackageUrl
		if (projectData.deploymentTrHashes)
			projectModel.deploymentTrHashes = JSON.parse(projectData.deploymentTrHashes)
		if (projectData.registerUrlTrHash)
			projectModel.registerUrlTrHash = projectData.registerUrlTrHash
		if (projectData.registerContentHashTrHash)
			projectModel.registerContentHashTrHash = projectData.registerContentHashTrHash
		if (projectData.registerContentHashBlockNumber)
			projectModel.registerContentHashBlockNumber = projectData.registerContentHashBlockNumber
		if (projectData.registerUrlBlockNumber)
			projectModel.registerUrlBlockNumber = projectData.registerUrlBlockNumber
		if (!projectData.title) {
			var parts = path.split("/");
			projectData.title = parts[parts.length - 2];
		}
		deploymentAddresses = projectData.deploymentAddresses ? projectData.deploymentAddresses : {};
		projectTitle = projectData.title;
		projectPath = path;
		if (!projectData.files)
			projectData.files = [];

		for(var i = 0; i < projectData.files.length; i++) {
			var entry = projectData.files[i];
			if (typeof(entry) === "string")
				addFile(entry); //TODO: remove old project file support
			else
				addFile(entry.fileName, entry.title);
		}
		if (mainApplication.trackLastProject)
			projectSettings.lastProjectPath = path;
		projectLoading(projectData);
		projectLoaded()

		//TODO: move this to codemodel
		var contractSources = {};
		for (var d = 0; d < listModel.count; d++) {
			var doc = listModel.get(d);
			if (doc.isContract)
				contractSources[doc.documentId] = fileIo.readFile(doc.path);
		}
		codeModel.reset(contractSources);
	});
}

function addFile(fileName, title) {
	var p = projectPath + fileName;
	var extension = fileName.substring(fileName.lastIndexOf("."), fileName.length);
	var isContract = extension === ".sol";
	var isHtml = extension === ".html";
	var isCss = extension === ".css";
	var isJs = extension === ".js";
	var isImg = extension === ".png"  || extension === ".gif" || extension === ".jpg" || extension === ".svg";
	var syntaxMode = isContract ? "solidity" : isJs ? "javascript" : isHtml ? "htmlmixed" : isCss ? "css" : "";
	var groupName = isContract ? qsTr("Contracts") : isJs ? qsTr("Javascript") : isHtml ? qsTr("Web Pages") : isCss ? qsTr("Styles") : isImg ? qsTr("Images") : qsTr("Misc");
	var docData = {
		contract: false,
		path: p,
		fileName: fileName,
		name: title !== undefined ? title : fileName,
									documentId: fileName,
									syntaxMode: syntaxMode,
									isText: isContract || isHtml || isCss || isJs,
									isContract: isContract,
									isHtml: isHtml,
									groupName: groupName
	};

	projectListModel.append(docData);
	fileIo.watchFileChanged(p);
	return docData.documentId;
}

function getDocumentIndex(documentId)
{
	for (var i = 0; i < projectListModel.count; i++)
		if (projectListModel.get(i).documentId === documentId)
			return i;
	console.error("Can't find document " + documentId);
	return -1;
}

function getDocumentByPath(_path)
{
	for (var i = 0; i < projectListModel.count; i++)
	{
		var doc = projectListModel.get(i);
		if (doc.path.indexOf(_path) !== -1)
			return doc.documentId;
	}
	return null;
}

function openDocument(documentId) {
	if (documentId !== currentDocumentId) {
		documentOpened(projectListModel.get(getDocumentIndex(documentId)));
		currentDocumentId = documentId;
	}
}

function openNextDocument() {
	var docIndex = getDocumentIndex(currentDocumentId);
	var nextDocId = "";
	while (nextDocId === "") {
		docIndex++;
		if (docIndex >= projectListModel.count)
			docIndex = 0;
		var document = projectListModel.get(docIndex);
		if (document.isText)
			nextDocId = document.documentId;
	}
	openDocument(nextDocId);
}

function openPrevDocument() {
	var docIndex = getDocumentIndex(currentDocumentId);
	var prevDocId = "";
	while (prevDocId === "") {
		docIndex--;
		if (docIndex < 0)
			docIndex = projectListModel.count - 1;
		var document = projectListModel.get(docIndex);
		if (document.isText)
			prevDocId = document.documentId;
	}
	openDocument(prevDocId);
}

function doCloseProject() {
	console.log("Closing project");
	projectListModel.clear();
	projectPath = "";
	currentDocumentId = "";
	projectClosed();
}

function doCreateProject(title, path) {
	closeProject(function() {
		console.log("Creating project " + title + " at " + path);
		if (path[path.length - 1] !== "/")
			path += "/";
		var dirPath = path + title + "/";
		fileIo.makeDir(dirPath);
		var projectFile = dirPath + projectFileName;

		var indexFile = "index.html";
		var contractsFile = "contract.sol";
		var projectData = {
			title: title,
			files: [ contractsFile, indexFile ]
		};
		//TODO: copy from template
		if (!fileIo.fileExists(dirPath + indexFile))
			fileIo.writeFile(dirPath + indexFile, htmlTemplate);
		if (!fileIo.fileExists(dirPath + contractsFile))
			fileIo.writeFile(dirPath + contractsFile, contractTemplate);
		newProject(projectData);
		var json = JSON.stringify(projectData, null, "\t");
		fileIo.writeFile(projectFile, json);
		loadProject(dirPath);
	});
}

function doAddExistingFiles(files) {
	for(var i = 0; i < files.length; i++) {
		var sourcePath = files[i];
		var sourceFileName = sourcePath.substring(sourcePath.lastIndexOf("/") + 1, sourcePath.length);
		var destPath = projectPath + sourceFileName;
		if (sourcePath !== destPath)
			fileIo.copyFile(sourcePath, destPath);
		var id = addFile(sourceFileName);
		saveProjectFile();
		documentAdded(id)
	}
}

function renameDocument(documentId, newName) {
	var i = getDocumentIndex(documentId);
	var document = projectListModel.get(i);
	if (!document.isContract) {
		fileIo.stopWatching(document.path);
		var sourcePath = document.path;
		var destPath = projectPath + newName;
		fileIo.moveFile(sourcePath, destPath);
		document.path = destPath;
		document.name = newName;
		document.fileName = newName;
		projectListModel.set(i, document);
		fileIo.watchFileChanged(destPath);
		saveProjectFile();
		documentUpdated(documentId);
	}
}

function getDocument(documentId) {
	var i = getDocumentIndex(documentId);
	if (i === -1)
		return undefined;
	else
		return projectListModel.get(i);
}

function getDocumentIdByName(fileName)
{
	for (var i = 0; i < projectListModel.count; i++)
		if (projectListModel.get(i).fileName === fileName)
			return projectListModel.get(i).documentId;
	return null;
}

function removeDocument(documentId) {
	var i = getDocumentIndex(documentId);
	var document = projectListModel.get(i);
	fileIo.stopWatching(document.path);
	fileIo.deleteFile(document.path);
	if (document.isContract)
		codeModel.unregisterContractSrc(documentId);
	projectListModel.remove(i);
	saveProjectFile();
	documentRemoved(documentId);
}

function newHtmlFile() {
	createAndAddFile("page", "html", htmlTemplate);
}

function newCssFile() {
	createAndAddFile("style", "css", "body {\n}\n");
}

function newJsFile() {
	createAndAddFile("script", "js", "function foo() {\n}\n");
}

function newContract() {
	createAndAddFile("contract", "sol", contractTemplate);
}


function createAndAddFile(name, extension, content) {
	var fileName = generateFileName(name, extension);
	var filePath = projectPath + fileName;
	if (!fileIo.fileExists(filePath))
		fileIo.writeFile(filePath, content);
	var id = addFile(fileName);
	saveProjectFile();
	documentAdded(id);
}

function generateFileName(name, extension) {
	var i = 1;
	do {
		var fileName = name + i + "." + extension;
		var filePath = projectPath + fileName;
		i++;
	} while (fileIo.fileExists(filePath));
	return fileName
}

