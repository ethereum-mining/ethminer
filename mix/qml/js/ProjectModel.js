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
Qt.include("QEtherHelper.js")

var htmlTemplate = "<html>\n<head>\n<script>\n</script>\n</head>\n<body>\n<script>\n</script>\n</body>\n</html>";
var contractTemplate = "contract Contract {\n}\n";

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
			deploymentAddresses: deploymentAddresses,
			applicationUrlEth: deploymentDialog.applicationUrlEth,
			applicationUrlHttp: deploymentDialog.applicationUrlHttp,
			packageHash: deploymentDialog.packageHash,
			packageBase64: deploymentDialog.packageBase64
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
	if (projectData.packageHash)
		deploymentDialog.packageHash =  projectData.packageHash
	if (projectData.packageBase64)
		deploymentDialog.packageBase64 =  projectData.packageBase64
	if (projectData.applicationUrlEth)
		deploymentDialog.applicationUrlEth = projectData.applicationUrlEth
	if (projectData.applicationUrlHttp)
		deploymentDialog.applicationUrlHttp = projectData.applicationUrlHttp
	if (!projectData.title) {
		var parts = path.split("/");
		projectData.title = parts[parts.length - 2];
	}
	deploymentAddresses = projectData.deploymentAddresses ? projectData.deploymentAddresses : [];
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

	//TODO: move this to codemodel
	var contractSources = {};
	for (var d = 0; d < listModel.count; d++) {
		var doc = listModel.get(d);
		if (doc.isContract)
			contractSources[doc.documentId] = fileIo.readFile(doc.path);
	}
	codeModel.reset(contractSources);

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
		name: fileName,
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
	console.log("Closing project");
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
	var contractsFile = "contract.sol";
	var projectData = {
		title: title,
		files: [ contractsFile, indexFile ]
	};
	//TODO: copy from template
	fileIo.writeFile(dirPath + indexFile, htmlTemplate);
	fileIo.writeFile(dirPath + contractsFile, contractTemplate);
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
	deploymentDialog.open();
}

function startDeployProject(erasePrevious)
{
	var date = new Date();
	var deploymentId = date.toLocaleString(Qt.locale(), "ddMMyyHHmmsszzz");
	if (!erasePrevious)
	{
		finalizeDeployment(deploymentId, projectModel.deploymentAddresses);
		return;
	}

	var jsonRpcUrl = "http://localhost:8080";
	console.log("Deploying " + deploymentId + " to " + jsonRpcUrl);
	deploymentStarted();

	var requests = [];
	var requestNames = [];

	for (var c in codeModel.contracts) { //TODO: order based on dependencies
		var code = codeModel.contracts[c].codeHex;
		requests.push({
			jsonrpc: "2.0",
			method: "eth_transact",
			params: [ { "code": code } ],
			id: jsonRpcRequestId++
		});
		requestNames.push(c);
	}

	var rpcRequest = JSON.stringify(requests);
	var httpRequest = new XMLHttpRequest();
	httpRequest.open("POST", jsonRpcUrl, true);
	httpRequest.setRequestHeader("Content-type", "application/json");
	httpRequest.setRequestHeader("Content-length", rpcRequest.length);
	httpRequest.setRequestHeader("Connection", "close");
	httpRequest.onreadystatechange = function() {
		if (httpRequest.readyState === XMLHttpRequest.DONE) {
			if (httpRequest.status === 200) {
				var rpcResponse = JSON.parse(httpRequest.responseText);
				if (rpcResponse.length === requestNames.length) {
					var contractAddresses = {};
					for (var r = 0; r < rpcResponse.length; r++)
						contractAddresses[requestNames[r]] = rpcResponse[r].result;
					finalizeDeployment(deploymentId, contractAddresses);
				}
			} else {
				var errorText = qsTr("Deployment error: RPC server HTTP status ") + httpRequest.status;
				console.log(errorText);
				deploymentError(errorText);
			}
		}
	}
	httpRequest.send(rpcRequest);
}

function finalizeDeployment(deploymentId, addresses) {
	deploymentStepChanged(qsTr("Packaging application ..."));
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
	var deploymentJs =
		"// Autogenerated by Mix\n" +
		"web3 = require(\"web3\");\n" +
		"contracts = {};\n";
	for (var c in codeModel.contracts)  {
		var contractAccessor = "contracts[\"" + codeModel.contracts[c].contract.name + "\"]";
		deploymentJs += contractAccessor + " = {\n" +
		"\tinterface: " + codeModel.contracts[c].contractInterface + ",\n" +
		"\taddress: \"" + addresses[c] + "\"\n" +
		"};\n" +
		contractAccessor + ".contract = web3.eth.contract(" + contractAccessor + ".address, " + contractAccessor + ".interface);\n";
	}
	fileIo.writeFile(deploymentDir + "deployment.js", deploymentJs);
	//copy scripts
	fileIo.copyFile("qrc:///js/bignumber.min.js", deploymentDir + "bignumber.min.js");
	fileIo.copyFile("qrc:///js/webthree.js", deploymentDir + "ethereum.js");
	deploymentAddresses = addresses;
	saveProject();

	var packageRet = fileIo.makePackage(deploymentDir);
	deploymentDialog.packageHash = packageRet[0];
	deploymentDialog.packageBase64 = packageRet[1];

	var applicationUrlEth = deploymentDialog.applicationUrlEth;
	applicationUrlEth = formatAppUrl(applicationUrlEth);

	deploymentStepChanged(qsTr("Registering application on the Ethereum network ..."));
	checkRegistration(applicationUrlEth, deploymentDialog.root, function () {
		deploymentComplete();
		deployRessourcesDialog.text = qsTr("Register Web Application to finalize deployment.");
		deployRessourcesDialog.open();
	});
}

function rpcCall(requests, callBack)
{
	var jsonRpcUrl = "http://localhost:8080";
	var rpcRequest = JSON.stringify(requests);
	var httpRequest = new XMLHttpRequest();
	httpRequest.open("POST", jsonRpcUrl, true);
	httpRequest.setRequestHeader("Content-type", "application/json");
	httpRequest.setRequestHeader("Content-length", rpcRequest.length);
	httpRequest.setRequestHeader("Connection", "close");
	httpRequest.onreadystatechange = function() {
		if (httpRequest.readyState === XMLHttpRequest.DONE) {
			if (httpRequest.status !== 200)
			{
				var errorText = qsTr("Deployment error: Error while registering Dapp ") + httpRequest.status;
				console.log(errorText);
				deploymentError(errorText);
				return;
			}
			callBack(httpRequest.status, httpRequest.responseText)
		}
	}
	httpRequest.send(rpcRequest);
}


function checkRegistration(dappUrl, addr, callBack)
{
	var requests = [];
	var data  = "";
	if (dappUrl.length > 0)
	{
		//checking path (register).
		var str = createString(dappUrl[0]);
		data  = "0x6be16bed" + str.encodeValueAsString();
		console.log("checking if path exists (register) => " + JSON.stringify(dappUrl));
		requests.push({
			jsonrpc: "2.0",
			method: "eth_call",
			params: [ { "to": '0x' + addr, "data": data } ],
			id: jsonRpcRequestId++
		});

		rpcCall(requests, function (httpRequest, response) {
			var address = JSON.parse(response)[0].result.replace('0x', '');
			if (address === "")
			{
				var errorTxt = qsTr("Path does not exists " + JSON.stringify(dappUrl) + " cannot continue");
				deploymentError(errorTxt);
				console.log(errorTxt);
				return;
			}

			dappUrl.splice(0, 1);
			checkRegistration(dappUrl, address, callBack);
		});
	}
	else
	{
		var paramTitle = createString(projectModel.projectTitle);
		requests.push({
						  //owner()
						  jsonrpc: "2.0",
						  method: "eth_call",
						  params: [ { "to": '0x' + addr, "data": "0xec7b9200" + paramTitle.encodeValueAsString() } ],
						  id: jsonRpcRequestId++
					  });

		requests.push({
						  //accounts
						  jsonrpc: "2.0",
						  method: "eth_accounts",
						  params: null,
						  id: jsonRpcRequestId++
					  });

		rpcCall(requests, function (httpRequest, response) {
			requests = [];
			var res = JSON.parse(response);
			var currentOwner = res[0].result;
			var noOwner = currentOwner.replace('0x', '').replace(/0/g, '') === '';

			if (noOwner)
			{
				requests.push({
							  //reserve()
							  jsonrpc: "2.0",
							  method: "eth_transact",
							  params: [ { "to": '0x' + addr, "data": "0x1c83171b" + paramTitle.encodeValueAsString() } ],
							  id: jsonRpcRequestId++
						  });
			}
			else
			{
				var bOwner = false;
				currentOwner = normalizeAddress(currentOwner);
				for (var u in res[1].result)
				{
					if (normalizeAddress(res[1].result[u]) === currentOwner)
						bOwner = true;
				}

				if (!bOwner)
				{
					var errorTxt = qsTr("Current user is not the owner of this path. Cannot continue")
					deploymentError(errorTxt);
					console.log(errorTxt);
					return;
				}
			}
			console.log("setContentHash");
			requests.push({
						  //setContent()
						  jsonrpc: "2.0",
						  method: "eth_transact",
						  params: [ { "to": '0x' + addr, "data": "0x5d574e32" + paramTitle.encodeValueAsString() + deploymentDialog.packageHash } ],
						  id: jsonRpcRequestId++
					  });
			rpcCall(requests, function (httpRequest, response) {
				callBack();
			});
		});
	}
}

function registerToUrlHint()
{
	deploymentStepChanged(qsTr("Registering application Resources (" + deploymentDialog.applicationUrlHttp) + ") ...");
	var requests = [];
	var paramUrlHttp = createString(deploymentDialog.applicationUrlHttp);
	requests.push({
				  //urlHint => suggestUrl
				  jsonrpc: "2.0",
				  method: "eth_transact",
				  params: [ { "to": '0x' + deploymentDialog.urlHintContract, "data": "0x4983e19c" + deploymentDialog.packageHash + paramUrlHttp.encodeValueAsString() } ],
				  id: jsonRpcRequestId++
			  });

	rpcCall(requests, function (httpRequest, response) {
		deploymentComplete();
	});
}

function normalizeAddress(addr)
{
	addr = addr.replace('0x', '');
	var i = 0;
	for (var k in addr)
	{
		if (addr[k] !== "0")
			break;
		else
			i++;
	}
	return addr.substring(i);
}

function formatAppUrl(url)
{
	var slash = url.indexOf("/");
	var dot = url.indexOf(".");
	if (slash === -1 && dot === -1)
		return url;
	if ((slash !== -1 && slash < dot) || dot === -1)
		return url.split("/");
	else
	{
		var dotted;
		var ret = [];
		if (slash !== -1)
		{
			ret.push(url.split("/"));
			dotted = ret[0].split(".");
		}
		else
			dotted = url.split(".");

		for (var k in dotted)
			ret.unshift(dotted[k]);
		return ret;
	}
}

