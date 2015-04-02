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
Qt.include("TransactionHelper.js")

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
			applicationUrlEth: deploymentDialog.applicationUrlEth,
			applicationUrlHttp: deploymentDialog.applicationUrlHttp,
			packageHash: deploymentDialog.packageHash,
			packageBase64: deploymentDialog.packageBase64,
			deploymentDir: projectModel.deploymentDir
		};
		for (var i = 0; i < projectListModel.count; i++)
			projectData.files.push(projectListModel.get(i).fileName);

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
			var projectData = JSON.parse(json);
			if (projectData.deploymentDir)
				projectModel.deploymentDir = projectData.deploymentDir
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
	});
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
		fileIo.writeFile(dirPath + indexFile, htmlTemplate);
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

	var jsonRpcUrl = "http://127.0.0.1:8080";
	console.log("Deploying " + deploymentId + " to " + jsonRpcUrl);
	deploymentStarted();

	var ctrNames = Object.keys(codeModel.contracts);
	var ctrAddresses = {};
	setDefaultBlock(0, function() {
		deployContracts(0, ctrAddresses, ctrNames, function (){
			finalizeDeployment(deploymentId, ctrAddresses);
		});
	});
}

function setDefaultBlock(val, callBack)
{
	var requests = [{
						jsonrpc: "2.0",
						method: "eth_setDefaultBlock",
						params: [val],
						id: 0
					}];
	rpcCall(requests, function (httpCall, response){
		callBack();
	});
}

function deployContracts(ctrIndex, ctrAddresses, ctrNames, callBack)
{
	var code = codeModel.contracts[ctrNames[ctrIndex]].codeHex;
	var requests = [{
						jsonrpc: "2.0",
						method: "eth_transact",
						params: [ { "from": deploymentDialog.currentAccount, "gas": deploymentDialog.gasToUse, "code": code } ],
						id: 0
					}];
	rpcCall(requests, function (httpCall, response){
		var txt = qsTr("Please wait while " + ctrNames[ctrIndex] + " is published ...")
		deploymentStepChanged(txt);
		console.log(txt);
		ctrAddresses[ctrNames[ctrIndex]] = JSON.parse(response)[0].result
		deploymentDialog.waitForTrCountToIncrement(function(status) {
			if (status === -1)
			{
				trCountIncrementTimeOut();
				return;
			}
			ctrIndex++;
			if (ctrIndex < ctrNames.length)
				deployContracts(ctrIndex, ctrAddresses, ctrNames, callBack);
			else
				callBack();
		});
	});
}

function finalizeDeployment(deploymentId, addresses) {
	deploymentStepChanged(qsTr("Packaging application ..."));
	var deploymentDir = projectPath + deploymentId + "/";
	projectModel.deploymentDir = deploymentDir;
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
	deploymentAddresses = addresses;
	saveProject();

	var packageRet = fileIo.makePackage(deploymentDir);
	deploymentDialog.packageHash = packageRet[0];
	deploymentDialog.packageBase64 = packageRet[1];

	var applicationUrlEth = deploymentDialog.applicationUrlEth;

	applicationUrlEth = formatAppUrl(applicationUrlEth);
	deploymentStepChanged(qsTr("Registering application on the Ethereum network ..."));
	checkEthPath(applicationUrlEth, function () {
		deploymentComplete();
		deployRessourcesDialog.text = qsTr("Register Web Application to finalize deployment.");
		deployRessourcesDialog.open();
		setDefaultBlock(-1, function() {});
	});
}

function checkEthPath(dappUrl, callBack)
{
	if (dappUrl.length === 1)
		registerContentHash(deploymentDialog.eth, callBack); // we directly create a dapp under the root registrar.
	else
	{
		// the first owned reigstrar must have been created to follow the path.
		var str = createString(dappUrl[0]);
		var requests = [];
		requests.push({
						  //register()
						  jsonrpc: "2.0",
						  method: "eth_call",
						  params: [ { "gas": 150000, "from": deploymentDialog.currentAccount,  "to": '0x' + deploymentDialog.eth, "data": "0x6be16bed"  + str.encodeValueAsString() } ],
						  id: jsonRpcRequestId++
					  });
		rpcCall(requests, function (httpRequest, response) {
			var res = JSON.parse(response);
			var addr = normalizeAddress(res[0].result);
			if (addr.replace(/0+/g, "") === "")
			{
				var errorTxt = qsTr("Path does not exists " + JSON.stringify(dappUrl) + ". Please register using Registration Dapp. Aborting.");
				deploymentError(errorTxt);
				console.log(errorTxt);
			}
			else
			{
				dappUrl.splice(0, 1);
				checkRegistration(dappUrl, addr, callBack);
			}
		});
	}
}

function checkRegistration(dappUrl, addr, callBack)
{
	if (dappUrl.length === 1)
		registerContentHash(addr, callBack); // We do not create the register for the last part, just registering the content hash.
	else
	{
		var txt = qsTr("Checking " + JSON.stringify(dappUrl) + " ... in registrar " + addr);
		deploymentStepChanged(txt);
		console.log(txt);
		var requests = [];
		var registrar = {}
		var str = createString(dappUrl[0]);
		requests.push({
						  //getOwner()
						  jsonrpc: "2.0",
						  method: "eth_call",
						  params: [ { "gas" : 2000, "from": deploymentDialog.currentAccount, "to": '0x' + addr, "data": "0x893d20e8" } ],
						  id: jsonRpcRequestId++
					  });

		requests.push({
						  //register()
						  jsonrpc: "2.0",
						  method: "eth_call",
						  params: [ { "from": deploymentDialog.currentAccount, "to": '0x' + addr, "data": "0x6be16bed"  + str.encodeValueAsString() } ],
						  id: jsonRpcRequestId++
					  });

		rpcCall(requests, function (httpRequest, response) {
			var res = JSON.parse(response);
			var nextAddr = normalizeAddress(res[1].result);
			var errorTxt;
			if (res[1].result === "0x")
			{
				errorTxt = qsTr("Error when creating new owned regsitrar. Please use the regsitration Dapp. Aborting");
				deploymentError(errorTxt);
				console.log(errorTxt);
			}
			else if (normalizeAddress(deploymentDialog.currentAccount) !== normalizeAddress(res[0].result))
			{
				errorTxt = qsTr("You are not the owner of " + dappUrl[0] + ". Aborting");
				deploymentError(errorTxt);
				console.log(errorTxt);
			}
			else if (nextAddr.replace(/0+/g, "") !== "")
			{
				dappUrl.splice(0, 1);
				checkRegistration(dappUrl, nextAddr, callBack);
			}
			else
			{
				var txt = qsTr("Registering sub domain " + dappUrl[0] +  " ...");
				console.log(txt);
				deploymentStepChanged(txt);
				//current registrar is owned => ownedregistrar creation and continue.
				requests = [];

				requests.push({
								  jsonrpc: "2.0",
								  method: "eth_transact",
								  params: [ { "from": deploymentDialog.currentAccount, "gas": 20000, "code": "0x60056013565b61059e8061001d6000396000f35b33600081905550560060003560e060020a90048063019848921461009a578063449c2090146100af5780635d574e32146100cd5780635fd4b08a146100e1578063618242da146100f65780636be16bed1461010b5780636c4489b414610129578063893d20e8146101585780639607730714610173578063c284bc2a14610187578063e50f599a14610198578063e5811b35146101af578063ec7b9200146101cd57005b6100a560043561031b565b8060005260206000f35b6100ba6004356103a0565b80600160a060020a031660005260206000f35b6100db600435602435610537565b60006000f35b6100ec600435610529565b8060005260206000f35b6101016004356103dd565b8060005260206000f35b6101166004356103bd565b80600160a060020a031660005260206000f35b61013460043561034b565b82600160a060020a031660005281600160a060020a03166020528060405260606000f35b610160610341565b80600160a060020a031660005260206000f35b6101816004356024356102b4565b60006000f35b6101926004356103fd565b60006000f35b6101a96004356024356044356101f2565b60006000f35b6101ba6004356101eb565b80600160a060020a031660005260206000f35b6101d8600435610530565b80600160a060020a031660005260206000f35b6000919050565b600054600160a060020a031633600160a060020a031614610212576102af565b8160026000858152602001908152602001600020819055508061023457610287565b81600160a060020a0316837f680ad70765443c2967675ab0fb91a46350c01c6df59bf9a41ff8a8dd097464ec60006000a3826001600084600160a060020a03168152602001908152602001600020819055505b827f18d67da0cd86808336a3aa8912f6ea70c5250f1a98b586d1017ef56fe199d4fc60006000a25b505050565b600054600160a060020a031633600160a060020a0316146102d457610317565b806002600084815260200190815260200160002060010181905550817f18d67da0cd86808336a3aa8912f6ea70c5250f1a98b586d1017ef56fe199d4fc60006000a25b5050565b60006001600083600160a060020a03168152602001908152602001600020549050919050565b6000600054905090565b6000600060006002600085815260200190815260200160002054925060026000858152602001908152602001600020600101549150600260008581526020019081526020016000206002015490509193909250565b600060026000838152602001908152602001600020549050919050565b600060026000838152602001908152602001600020600101549050919050565b600060026000838152602001908152602001600020600201549050919050565b600054600160a060020a031633600160a060020a03161461041d57610526565b80600160006002600085815260200190815260200160002054600160a060020a031681526020019081526020016000205414610458576104d2565b6002600082815260200190815260200160002054600160a060020a0316817f680ad70765443c2967675ab0fb91a46350c01c6df59bf9a41ff8a8dd097464ec60006000a36000600160006002600085815260200190815260200160002054600160a060020a03168152602001908152602001600020819055505b6002600082815260200190815260200160002060008101600090556001810160009055600281016000905550807f18d67da0cd86808336a3aa8912f6ea70c5250f1a98b586d1017ef56fe199d4fc60006000a25b50565b6000919050565b6000919050565b600054600160a060020a031633600160a060020a0316146105575761059a565b806002600084815260200190815260200160002060020181905550817f18d67da0cd86808336a3aa8912f6ea70c5250f1a98b586d1017ef56fe199d4fc60006000a25b505056" } ],
								  id: jsonRpcRequestId++
							  });

				rpcCall(requests, function(httpRequest, response) {
					var newCtrAddress = normalizeAddress(JSON.parse(response)[0].result);
					requests = [];
					var txt = qsTr("Please wait " + dappUrl[0] + " is registering ...");
					deploymentStepChanged(txt);
					console.log(txt);
					deploymentDialog.waitForTrCountToIncrement(function(status) {
						if (status === -1)
						{
							trCountIncrementTimeOut();
							return;
						}
						var crLevel = createString(dappUrl[0]).encodeValueAsString();
						requests.push({
										  //setRegister()
										  jsonrpc: "2.0",
										  method: "eth_transact",
										  params: [ { "from": deploymentDialog.currentAccount, "gas": 30000, "to": '0x' + addr, "data": "0x96077307" + crLevel + deploymentDialog.pad(newCtrAddress) } ],
										  id: jsonRpcRequestId++
									  });

						rpcCall(requests, function(request, response){
							dappUrl.splice(0, 1);
							checkRegistration(dappUrl, newCtrAddress, callBack);
						});
					});
				});
			}
		});
	}
}

function trCountIncrementTimeOut()
{
	var error = qsTr("Something went wrong during the deployment. Please verify the amount of gas for this transaction and check your balance.")
	console.log(error);
	deploymentError(error);
}

function registerContentHash(registrar, callBack)
{
	var txt = qsTr("Finalizing Dapp registration ...");
	deploymentStepChanged(txt);
	console.log(txt);
	var requests = [];
	var paramTitle = createString(projectModel.projectTitle);
	requests.push({
					  //setContent()
					  jsonrpc: "2.0",
					  method: "eth_transact",
					  params: [ { "from": deploymentDialog.currentAccount, "gas": 30000, "gasPrice": "10", "to": '0x' + registrar, "data": "0x5d574e32" + paramTitle.encodeValueAsString() + deploymentDialog.packageHash } ],
					  id: jsonRpcRequestId++
				  });
	rpcCall(requests, function (httpRequest, response) {
		callBack();
	});
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
					  params: [ {  "to": '0x' + deploymentDialog.urlHintContract, "gas": 30000, "data": "0x4983e19c" + deploymentDialog.packageHash + paramUrlHttp.encodeValueAsString() } ],
					  id: jsonRpcRequestId++
				  });

	rpcCall(requests, function (httpRequest, response) {
		deploymentComplete();
	});
}

function normalizeAddress(addr)
{
	addr = addr.replace('0x', '');
	if (addr.length <= 40)
		return addr;
	var left = addr.length - 40;
	return addr.substring(left);
}

function formatAppUrl(url)
{
	if (url.toLowerCase().indexOf("eth://") === 0)
		url = url.substring(6);
	if (url.toLowerCase().indexOf(projectModel.projectTitle + ".") === 0)
		url = url.substring(projectModel.projectTitle.length + 1);
	if (url === "")
		return [projectModel.projectTitle];

	var ret;
	if (url.indexOf("/") === -1)
		ret = url.split('.').reverse();
	else
	{
		var slash = url.indexOf("/");
		var left = url.substring(0, slash);
		var leftA = left.split(".");
		leftA.reverse();

		var right = url.substring(slash + 1);
		var rightA = right.split('/');
		ret = leftA.concat(rightA);
	}
	if (ret[0].toLowerCase() === "eth")
		ret.splice(0, 1);
	ret.push(projectModel.projectTitle);
	return ret;
}
