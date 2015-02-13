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
		var projectData = {
			files: [],
			title: projectTitle,
			deploymentAddress: deploymentAddress
		};
		for (var i = 0; i < projectListModel.count; i++)
			projectData.files.push(projectListModel.get(i).fileName)
		projectSaving(projectData);
		var json = JSON.stringify(projectData, null, "\t");
		var projectFile = projectPath + projectFileName;
		fileIo.writeFile(projectFile, json);
		projectSaved();
	}
}

function loadProject(path) {
	closeProject();
	console.log("loading project at " + path);
	var projectFile = path + projectFileName;
	var json = fileIo.readFile(projectFile);
	var projectData = JSON.parse(json);
	if (!projectData.title) {
		var parts = path.split("/");
		projectData.title = parts[parts.length - 2];
	}
	deploymentAddress = projectData.deploymentAddress ? projectData.deploymentAddress : "";
	projectTitle = projectData.title;
	projectPath = path;
	if (!projectData.files)
		projectData.files = [];

	for(var i = 0; i < projectData.files.length; i++) {
		addFile(projectData.files[i]);
	}
	projectSettings.lastProjectPath = path;
	projectLoading(projectData);
	projectLoaded()
}

function addFile(fileName) {
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
		name: isContract ? "Contract" : fileName,
		documentId: fileName,
		syntaxMode: syntaxMode,
		isText: isContract || isHtml || isCss || isJs,
		isContract: isContract,
		isHtml: isHtml,
		groupName: groupName
	};

	projectListModel.append(docData);
	return docData.documentId;
}

function getDocumentIndex(documentId)
{
	for (var i = 0; i < projectListModel.count; i++)
		if (projectListModel.get(i).documentId === documentId)
			return i;
	console.error("Cant find document " + documentId);
	return -1;
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
	console.log("closing project");
	projectListModel.clear();
	projectPath = "";
	currentDocumentId = "";
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

	var indexFile = "index.html";
	var contractsFile = "contracts.sol";
	var projectData = {
		title: title,
		files: [ contractsFile, indexFile ]
	};
	//TODO: copy from template
	fileIo.writeFile(dirPath + indexFile, "<html>\n<head>\n<script>\nvar web3 = parent.web3;\nvar theContract = parent.contract;\n</script>\n</head>\n<body>\n<script>\n</script>\n</body>\n</html>");
	fileIo.writeFile(dirPath + contractsFile, "contract Contract {\n}\n");
	newProject(projectData);
	var json = JSON.stringify(projectData, null, "\t");
	fileIo.writeFile(projectFile, json);
	loadProject(dirPath);
}

function doAddExistingFiles(files) {
	for(var i = 0; i < files.length; i++) {
		var sourcePath = files[i];
		var sourceFileName = sourcePath.substring(sourcePath.lastIndexOf("/") + 1, sourcePath.length);
		var destPath = projectPath + sourceFileName;
		if (sourcePath !== destPath)
			fileIo.copyFile(sourcePath, destPath);
		var id = addFile(sourceFileName);
		documentAdded(id)
	}
}

function renameDocument(documentId, newName) {
	var i = getDocumentIndex(documentId);
	var document = projectListModel.get(i);
	if (!document.isContract) {
		var sourcePath = document.path;
		var destPath = projectPath + newName;
		fileIo.moveFile(sourcePath, destPath);
		document.path = destPath;
		document.name = newName;
		projectListModel.set(i, document);
		documentUpdated(documentId);
	}
}

function getDocument(documentId) {
	var i = getDocumentIndex(documentId);
	return projectListModel.get(i);
}

function removeDocument(documentId) {
	var i = getDocumentIndex(documentId);
	var document = projectListModel.get(i);
	if (!document.isContract) {
		projectListModel.remove(i);
		documentRemoved(documentId);
	}
}

function newHtmlFile() {
	createAndAddFile("page", "html", "<html>\n</html>");
}

function newCssFile() {
	createAndAddFile("style", "css", "body {\n}\n");
}

function newJsFile() {
	createAndAddFile("script", "js", "function foo() {\n}\n");
}

function createAndAddFile(name, extension, content) {
	var fileName = generateFileName(name, extension);
	var filePath = projectPath + fileName;
	fileIo.writeFile(filePath, content);
	var id = addFile(fileName);
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


var jsonRpcRequestId = 1;
function deployProject(force) {

	saveAll(); //TODO: ask user

	if (!force && deploymentAddress !== "") {
		deployWarningDialog.visible = true;
		return;
	}

	var date = new Date();
	var deploymentId = date.toLocaleString(Qt.locale(), "ddMMyyHHmmsszzz");
	var jsonRpcUrl = "http://localhost:8080";
	console.log("Deploying " + deploymentId + " to " + jsonRpcUrl);
	deploymentStarted();
	var code = codeModel.codeHex
	var rpcRequest = JSON.stringify({
		jsonrpc: "2.0",
		method: "eth_transact",
		params: [ {
				"code": code
			} ],
		id: jsonRpcRequestId++
	});
	var httpRequest = new XMLHttpRequest()
	httpRequest.open("POST", jsonRpcUrl, true);
	httpRequest.setRequestHeader("Content-type", "application/json");
	httpRequest.setRequestHeader("Content-length", rpcRequest.length);
	httpRequest.setRequestHeader("Connection", "close");
	httpRequest.onreadystatechange = function() {
		if (httpRequest.readyState === XMLHttpRequest.DONE) {
			if (httpRequest.status === 200) {
				var rpcResponse = JSON.parse(httpRequest.responseText);
				var address = rpcResponse.result;
				console.log("Created contract, address: " + address);
				finalizeDeployment(deploymentId, address);
			} else {
				var errorText = qsTr("Deployment error: RPC server HTTP status ") + httpRequest.status;
				console.log(errorText);
				deploymentError(errorText);
			}
		}
	}
	httpRequest.send(rpcRequest);
}

function finalizeDeployment(deploymentId, address) {
	//create a dir for frontend files and copy them
	var deploymentDir = projectPath + deploymentId + "/";
	fileIo.makeDir(deploymentDir);
	for (var i = 0; i < projectListModel.count; i++) {
		var doc = projectListModel.get(i);
		if (doc.isContract)
			continue;
		if (doc.isHtml) {
			//inject the script to access contract API
			//TODO: use a template
			var html = fileIo.readFile(doc.path);
			var insertAt = html.indexOf("<head>")
			if (insertAt < 0)
				insertAt = 0;
			else
				insertAt += 6;
			html = html.substr(0, insertAt) +
					"<script src=\"bignumber.min.js\"></script>" +
					"<script src=\"ethereum.js\"></script>" +
					"<script src=\"deployment.js\"></script>" +
					html.substr(insertAt);
			fileIo.writeFile(deploymentDir + doc.fileName, html);
		}
		else
			fileIo.copyFile(doc.path, deploymentDir + doc.fileName);
	}
	//write deployment js
	var contractAccessor = "contracts[\"" + codeModel.code.contract.name + "\"]";
	var deploymentJs =
		"// Autogenerated by Mix\n" +
		"web3 = require(\"web3\");\n" +
		"contracts = {};\n" +
		contractAccessor + " = {\n" +
		"\tinterface: " + codeModel.code.contractInterface + ",\n" +
		"\taddress: \"" + address + "\"\n" +
		"};\n" +
		contractAccessor + ".contract = web3.eth.contract(" + contractAccessor + ".address, " + contractAccessor + ".interface);\n";
	fileIo.writeFile(deploymentDir + "deployment.js", deploymentJs);
	//copy scripts
	fileIo.copyFile("qrc:///js/bignumber.min.js", deploymentDir + "bignumber.min.js");
	fileIo.copyFile("qrc:///js/webthree.js", deploymentDir + "ethereum.js");
	deploymentAddress = address;
	saveProject();
	deploymentComplete();
}
