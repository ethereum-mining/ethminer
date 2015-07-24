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
/** @file NetworkDeployment.js
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @author Yann yann@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */
.import org.ethereum.qml.QSolidityType 1.0 as QSolidityType

Qt.include("TransactionHelper.js")
Qt.include("QEtherHelper.js")


var jsonRpcRequestId = 1;
function deployProject(force) {
	saveAll(); //TODO: ask user
	deploymentDialog.open();
}

function deployContracts(gas, gasPrice, callback)
{	
	deploymentGas = gas;
	deploymentGasPrice = gasPrice
	deploymentStarted();

	var ctrAddresses = {};
	var state = retrieveState(projectModel.deployedScenarioIndex);
	if (!state)
	{
		var txt = qsTr("Unable to find this scenario");
		deploymentError(txt);
		console.log(txt);
		return;
	}
	var trHashes = {}
	executeTr(0, 0, state, ctrAddresses, trHashes, function(){
		projectModel.deploymentAddresses = ctrAddresses;
		deploymentStepChanged(qsTr("Scenario deployed. Please wait for verifications"))
		if (callback)
			callback(ctrAddresses, trHashes)
	});
}

function checkPathCreationCost(ethUrl, callBack)
{
	var dappUrl = formatAppUrl(ethUrl);
	checkEthPath(dappUrl, true, function(success, cause) {
		if (!success)
		{
			switch (cause)
			{
			case "rootownedregistrar_notexist":
				deploymentError(qsTr("Owned registrar does not exist under the global registrar. Please create one using DApp registration."));
				break;
			case "ownedregistrar_creationfailed":
				deploymentError(qsTr("The creation of your new owned registrar fails. Please use DApp registration to create one."));
				break;
			case "ownedregistrar_notowner":
				deploymentError(qsTr("You are not the owner of this registrar. You cannot register your Dapp here."));
				break;
			default:
				break;
			}
		}
		else
			callBack((dappUrl.length - 1) * (deploymentDialog.registerStep.ownedRegistrarDeployGas + deploymentDialog.registerStep.ownedRegistrarSetSubRegistrarGas) + deploymentDialog.registerStep.ownedRegistrarSetContentHashGas);
	});
}

function gasUsed()
{
	var gas = 0;
	var gasCosts = clientModel.gasCosts;
	for (var g in gasCosts)
	{
		gas += gasCosts[g];
	}
	return gas;
}

function retrieveState(stateIndex)
{
	return projectModel.stateListModel.get(stateIndex);
}

function replaceParamToken(paramsDef, params, ctrAddresses)
{
	var retParams = {};
	for (var k in paramsDef)
	{
		var value = "";
		if (params[paramsDef[k].name] !== undefined)
		{
			value = params[paramsDef[k].name];
			if (paramsDef[k].type.category === 4 && value.indexOf("<") === 0)
			{
				value = value.replace("<", "").replace(">", "");
				value = ctrAddresses[value];
			}
		}
		retParams[paramsDef[k].name] = value;
	}
	return retParams;
}

function getFunction(ctrName, functionId)
{
	ctrName = contractFromToken(ctrName)
	if (codeModel.contracts[ctrName] === undefined)
		return null;
	if (ctrName === functionId)
		return codeModel.contracts[ctrName].contract.constructor;
	else
	{
		for (var j in codeModel.contracts[ctrName].contract.functions)
		{
			if (codeModel.contracts[ctrName].contract.functions[j].name === functionId)
				return codeModel.contracts[ctrName].contract.functions[j];
		}
	}
}

var deploymentGas
var deploymentGasPrice
var trRealIndex = -1
function executeTr(blockIndex, trIndex, state, ctrAddresses, trHashes, callBack)
{
	trRealIndex++;
	var tr = state.blocks.get(blockIndex).transactions.get(trIndex);
	if (!tr)
		callBack()
	var func = getFunction(tr.contractId, tr.functionId);
	if (!func)
		executeTrNextStep(blockIndex, trIndex, state, ctrAddresses, trHashes, callBack);
	else
	{
		var gasCost = clientModel.toHex(deploymentGas[trRealIndex]);
		var rpcParams = { "from": deploymentDialog.worker.currentAccount, "gas": "0x" + gasCost, "gasPrice": deploymentGasPrice };
		var params = replaceParamToken(func.parameters, tr.parameters, ctrAddresses);
		var encodedParams = clientModel.encodeParams(params, contractFromToken(tr.contractId), tr.functionId);

		if (tr.contractId === tr.functionId)
			rpcParams.code = codeModel.contracts[tr.contractId].codeHex + encodedParams.join("");
		else
		{
			rpcParams.data = "0x" + func.qhash() + encodedParams.join("");
			rpcParams.to = ctrAddresses[tr.contractId];
		}
		
		var requests = [{
							jsonrpc: "2.0",
							method: "eth_sendTransaction",
							params: [ rpcParams ],
							id: jsonRpcRequestId
						}];

		rpcCall(requests, function (httpCall, response){
			var txt = qsTr(tr.contractId + "." + tr.functionId + "() ...")
			deploymentStepChanged(txt);
			console.log(txt);
			var hash = JSON.parse(response)[0].result
			trHashes[tr.contractId + "." + tr.functionId + "()"] = hash
			deploymentDialog.worker.waitForTrReceipt(hash, function(status, receipt){
				if (status === -1)
					trCountIncrementTimeOut();
				else
				{
					if (tr.contractId === tr.functionId)
					{
						ctrAddresses[tr.contractId] = receipt.contractAddress
						ctrAddresses["<" + tr.contractId + " - " + trIndex  + ">"] = receipt.contractAddress //get right ctr address if deploy more than one contract of same type.
					}
					executeTrNextStep(blockIndex, trIndex, state, ctrAddresses, trHashes, callBack)
				}
			})
		});
	}
}

function retrieveContractAddress(trHash, callback)
{
	var requests = [{
						jsonrpc: "2.0",
						method: "eth_getTransactionReceipt",
						params: [ trHash ],
						id: jsonRpcRequestId
					}];
	rpcCall(requests, function (httpCall, response){
		callback(JSON.parse(response)[0].contractAddress)
	})
}

function executeTrNextStep(blockIndex, trIndex, state, ctrAddresses, trHashes, callBack)
{
	trIndex++;
	if (trIndex < state.blocks.get(blockIndex).transactions.count)
		executeTr(blockIndex, trIndex, state, ctrAddresses, trHashes, callBack);
	else
	{
		blockIndex++
		if (blockIndex < state.blocks.count)
			executeTr(blockIndex, 0, state, ctrAddresses, trHashes, callBack);
		else
			callBack();
	}
}

function gasPrice(callBack, error)
{
	var requests = [{
						jsonrpc: "2.0",
						method: "eth_gasPrice",
						params: [],
						id: jsonRpcRequestId
					}];
	rpcCall(requests, function (httpCall, response){
		callBack(JSON.parse(response)[0].result)
	}, function(message){
		error(message)
	});
}

function packageDapp(addresses)
{
	var date = new Date();
	var deploymentId = date.toLocaleString(Qt.locale(), "ddMMyyHHmmsszzz");
	deploymentDialog.packageStep.deploymentId = deploymentId

	deploymentStepChanged(qsTr("Packaging application ..."));
	var deploymentDir = projectPath + deploymentId + "/";

	if (deploymentDialog.packageStep.packageDir !== "")
		deploymentDir = deploymentDialog.packageStep.packageDir
	else
		deploymentDir = projectPath + "package/"

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
				contractAccessor + ".contractClass = web3.eth.contract(" + contractAccessor + ".interface);\n" +
				contractAccessor + ".contract = " + contractAccessor + ".contractClass.at(" + contractAccessor + ".address);\n";
	}
	fileIo.writeFile(deploymentDir + "deployment.js", deploymentJs);
	deploymentAddresses = addresses;
	saveProject();

	var packageRet = fileIo.makePackage(deploymentDir);
	deploymentDialog.packageStep.packageHash = packageRet[0];
	deploymentDialog.packageStep.packageBase64 = packageRet[1];
	deploymentDialog.packageStep.localPackageUrl = packageRet[2] + "?hash=" + packageRet[0];
	deploymentDialog.packageStep.lastDeployDate = date
	deploymentStepChanged(qsTr("Dapp is Packaged"))
}

function registerDapp(url, gasPrice, callback)
{
	deploymentGasPrice = gasPrice
	deploymentStepChanged(qsTr("Registering application on the Ethereum network ..."));
	checkEthPath(url, false, function (success) {
		if (!success)
			return;
		deploymentStepChanged(qsTr("Dapp has been registered. Please wait for verifications."));
		if (callback)
			callback()
	});
}

function checkEthPath(dappUrl, checkOnly, callBack)
{
	if (dappUrl.length === 1)
	{
		// convenient for dev purpose, should not be possible in normal env.
		if (!checkOnly)
			reserve(deploymentDialog.registerStep.eth, function() {
				registerContentHash(deploymentDialog.registerStep.eth, callBack); // we directly create a dapp under the root registrar.
			});
		else
			callBack(true);
	}
	else
	{
		// the first owned registrar must have been created to follow the path.
		var str = clientModel.encodeStringParam(dappUrl[0]);
		var requests = [];
		requests.push({
						  //subRegistrar()
						  jsonrpc: "2.0",
						  method: "eth_call",
						  params: [ { "gas": "0xffff", "from": deploymentDialog.worker.currentAccount,  "to": '0x' + deploymentDialog.registerStep.eth, "data": "0xe1fa8e84" + str }, "pending" ],
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
				callBack(false, "rootownedregistrar_notexist");
			}
			else
			{
				dappUrl.splice(0, 1);
				checkRegistration(dappUrl, addr, callBack, checkOnly);
			}
		});
	}
}

function isOwner(addr, callBack)
{
	var requests = [];
	requests.push({
					  //getOwner()
					  jsonrpc: "2.0",
					  method: "eth_call",
					  params: [ { "from": deploymentDialog.worker.currentAccount, "to": '0x' + addr, "data": "0xb387ef92" }, "pending" ],
					  id: jsonRpcRequestId++
				  });
	rpcCall(requests, function (httpRequest, response) {
		var res = JSON.parse(response);
		callBack(normalizeAddress(deploymentDialog.worker.currentAccount) === normalizeAddress(res[0].result));
	});
}

function checkRegistration(dappUrl, addr, callBack, checkOnly)
{
	isOwner(addr, function(ret){
		if (!ret)
		{
			var errorTxt = qsTr("You are not the owner of " + dappUrl[0] + ". Aborting");
			deploymentError(errorTxt);
			console.log(errorTxt);
			callBack(false, "ownedregistrar_notowner");
		}
		else
			continueRegistration(dappUrl, addr, callBack, checkOnly);
	});
}

function continueRegistration(dappUrl, addr, callBack, checkOnly)
{
	if (dappUrl.length === 1)
	{
		if (!checkOnly)
			registerContentHash(addr, callBack); // We do not create the register for the last part, just registering the content hash.
		else
			callBack(true);
	}
	else
	{
		var txt = qsTr("Checking " + JSON.stringify(dappUrl));
		deploymentStepChanged(txt);
		console.log(txt);
		var requests = [];
		var registrar = {}
		var str = clientModel.encodeStringParam(dappUrl[0]);


		requests.push({
						  //register()
						  jsonrpc: "2.0",
						  method: "eth_call",
						  params: [ { "from": deploymentDialog.worker.currentAccount, "to": '0x' + addr, "data": "0x5a3a05bd"  + str }, "pending" ],
						  id: jsonRpcRequestId++
					  });

		rpcCall(requests, function (httpRequest, response) {
			var res = JSON.parse(response);
			var nextAddr = normalizeAddress(res[0].result);
			var errorTxt;
			if (res[0].result === "0x")
			{
				errorTxt = qsTr("Error when creating new owned registrar. Please use the registration Dapp. Aborting");
				deploymentError(errorTxt);
				console.log(errorTxt);
				callBack(false, "ownedregistrar_creationfailed");
			}
			else if (nextAddr.replace(/0+/g, "") !== "")
			{
				dappUrl.splice(0, 1);
				checkRegistration(dappUrl, nextAddr, callBack, checkOnly);
			}
			else
			{
				if (checkOnly)
				{
					callBack(true);
					return;
				}
				var txt = qsTr("Registering sub domain " + dappUrl[0] +  " ...");
				console.log(txt);
				deploymentStepChanged(txt);
				//current registrar is owned => ownedregistrar creation and continue.
				requests = [];
				var gasCost = clientModel.toHex(deploymentDialog.registerStep.ownedRegistrarDeployGas);
				requests.push({
								  jsonrpc: "2.0",
								  method: "eth_sendTransaction",
								  params: [ { "from": deploymentDialog.worker.currentAccount, "gasPrice": deploymentGasPrice, "gas": "0x" + gasCost, "code": "0x600080547fffffffffffffffffffffffff000000000000000000000000000000000000000016331781556105cd90819061003990396000f3007c010000000000000000000000000000000000000000000000000000000060003504630198489281146100b257806321f8a721146100e45780632dff6941146100ee5780633b3b57de1461010e5780635a3a05bd1461013e5780635fd4b08a146101715780637dd564111461017d57806389a69c0e14610187578063b387ef92146101bb578063b5c645bd146101f4578063be99a98014610270578063c3d014d6146102a8578063d93e7573146102dc57005b73ffffffffffffffffffffffffffffffffffffffff600435166000908152600160205260409020548060005260206000f35b6000808052602081f35b600435600090815260026020819052604090912001548060005260206000f35b600435600090815260026020908152604082205473ffffffffffffffffffffffffffffffffffffffff1680835291f35b600435600090815260026020908152604082206001015473ffffffffffffffffffffffffffffffffffffffff1680835291f35b60008060005260206000f35b6000808052602081f35b60005461030c9060043590602435903373ffffffffffffffffffffffffffffffffffffffff908116911614610569576105c9565b60005473ffffffffffffffffffffffffffffffffffffffff168073ffffffffffffffffffffffffffffffffffffffff1660005260206000f35b600435600090815260026020819052604090912080546001820154919092015473ffffffffffffffffffffffffffffffffffffffff9283169291909116908273ffffffffffffffffffffffffffffffffffffffff166000528173ffffffffffffffffffffffffffffffffffffffff166020528060405260606000f35b600054610312906004359060243590604435903373ffffffffffffffffffffffffffffffffffffffff90811691161461045457610523565b6000546103189060043590602435903373ffffffffffffffffffffffffffffffffffffffff90811691161461052857610565565b60005461031e90600435903373ffffffffffffffffffffffffffffffffffffffff90811691161461032457610451565b60006000f35b60006000f35b60006000f35b60006000f35b60008181526002602090815260408083205473ffffffffffffffffffffffffffffffffffffffff16835260019091529020548114610361576103e1565b6000818152600260205260408082205473ffffffffffffffffffffffffffffffffffffffff169183917ff63780e752c6a54a94fc52715dbc5518a3b4c3c2833d301a204226548a2a85459190a360008181526002602090815260408083205473ffffffffffffffffffffffffffffffffffffffff16835260019091528120555b600081815260026020819052604080832080547fffffffffffffffffffffffff00000000000000000000000000000000000000009081168255600182018054909116905590910182905582917fa6697e974e6a320f454390be03f74955e8978f1a6971ea6730542e37b66179bc91a25b50565b600083815260026020526040902080547fffffffffffffffffffffffff00000000000000000000000000000000000000001683179055806104bb57827fa6697e974e6a320f454390be03f74955e8978f1a6971ea6730542e37b66179bc60006040a2610522565b73ffffffffffffffffffffffffffffffffffffffff8216837ff63780e752c6a54a94fc52715dbc5518a3b4c3c2833d301a204226548a2a854560006040a373ffffffffffffffffffffffffffffffffffffffff821660009081526001602052604090208390555b5b505050565b600082815260026020819052604080832090910183905583917fa6697e974e6a320f454390be03f74955e8978f1a6971ea6730542e37b66179bc91a25b5050565b60008281526002602052604080822060010180547fffffffffffffffffffffffff0000000000000000000000000000000000000000168417905583917fa6697e974e6a320f454390be03f74955e8978f1a6971ea6730542e37b66179bc91a25b505056" } ],
								  id: jsonRpcRequestId++
							  });

				rpcCall(requests, function(httpRequest, response) {
					var newCtrAddress = normalizeAddress(JSON.parse(response)[0].result);
					requests = [];
					var txt = qsTr("Please wait " + dappUrl[0] + " is registering ...");
					deploymentStepChanged(txt);
					console.log(txt);
					deploymentDialog.worker.waitForTrCountToIncrement(function(status) {
						if (status === -1)
						{
							trCountIncrementTimeOut();
							return;
						}
						var crLevel = clientModel.encodeStringParam(dappUrl[0]);
						var gasCost = clientModel.toHex(deploymentDialog.registerStep.ownedRegistrarSetSubRegistrarGas);
						requests.push({
										  //setRegister()
										  jsonrpc: "2.0",
										  method: "eth_sendTransaction",
										  params: [ { "from": deploymentDialog.worker.currentAccount, "gasPrice": deploymentGasPrice, "gas": "0x" + gasCost, "to": '0x' + addr, "data": "0x89a69c0e" + crLevel + newCtrAddress } ],
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

function reserve(registrar, callBack)
{
	var txt = qsTr("Making reservation in the root registrar...");
	deploymentStepChanged(txt);
	console.log(txt);
	var requests = [];
	var paramTitle = clientModel.encodeStringParam(projectModel.projectTitle);
	requests.push({
				  //reserve()
				  jsonrpc: "2.0",
				  method: "eth_sendTransaction",
				  params: [ { "from": deploymentDialog.worker.currentAccount, "gasPrice": deploymentGasPrice, "gas": "0xfffff", "to": '0x' + registrar, "data": "0x432ced04" + paramTitle } ],
				  id: jsonRpcRequestId++
			  });
	rpcCall(requests, function (httpRequest, response) {
		callBack();
	});
}


function registerContentHash(registrar, callBack)
{
	var txt = qsTr("Finalizing Dapp registration ...");
	deploymentStepChanged(txt);
	console.log(txt);
	console.log("register url " + deploymentDialog.packageStep.packageHash + " " + projectModel.projectTitle)
	projectModel.registerContentHashTrHash = ""
	var requests = [];
	var paramTitle = clientModel.encodeStringParam(projectModel.projectTitle);
	var gasCost = clientModel.toHex(deploymentDialog.registerStep.ownedRegistrarSetContentHashGas);
	requests.push({
					  //setContent()
					  jsonrpc: "2.0",
					  method: "eth_sendTransaction",
					  params: [ { "from": deploymentDialog.worker.currentAccount, "gasPrice": deploymentGasPrice, "gas": "0x" + gasCost, "to": '0x' + registrar, "data": "0xc3d014d6" + paramTitle + deploymentDialog.packageStep.packageHash } ],
					  id: jsonRpcRequestId++
				  });
	rpcCall(requests, function (httpRequest, response) {
		projectModel.registerContentHashTrHash = JSON.parse(response)[0].result
		callBack(true);
	});
}

function registerToUrlHint(url, gasPrice, callback)
{
	console.log("register url " + deploymentDialog.packageStep.packageHash + " " + url)
	deploymentGasPrice = gasPrice
	deploymentStepChanged(qsTr("Registering application Resources..."))
	urlHintAddress(function(urlHint){
		var requests = [];
		var paramUrlHttp = clientModel.encodeStringParam(url)
		var gasCost = clientModel.toHex(deploymentDialog.registerStep.urlHintSuggestUrlGas);
		requests.push({
						  //urlHint => suggestUrl
						  jsonrpc: "2.0",
						  method: "eth_sendTransaction",
						  params: [ {  "to": '0x' + urlHint, "gasPrice": deploymentGasPrice, "from": deploymentDialog.worker.currentAccount, "gas": "0x" + gasCost, "data": "0x584e86ad" + deploymentDialog.packageStep.packageHash + paramUrlHttp } ],
						  id: jsonRpcRequestId++
					  });

		rpcCall(requests, function (httpRequest, response) {
			projectModel.registerUrlTrHash = JSON.parse(response)[0].result
			deploymentStepChanged(qsTr("Dapp resources has been registered. Please wait for verifications."));
			if (callback)
				callback()
		});
	});
}

function urlHintAddress(callBack)
{
	var requests = [];
	var urlHint = clientModel.encodeStringParam("urlhint");
	requests.push({
					  //registrar: get UrlHint addr
					  jsonrpc: "2.0",
					  method: "eth_call",
					  params: [ {  "to": '0x' + deploymentDialog.registerStep.eth, "from": deploymentDialog.worker.currentAccount, "data": "0x3b3b57de" + urlHint }, "pending" ],
					  id: jsonRpcRequestId++
				  });

	rpcCall(requests, function (httpRequest, response) {
		var res = JSON.parse(response);
		callBack(normalizeAddress(res[0].result));
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
	if (!url)
		return [projectModel.projectTitle];
	if (url.toLowerCase().lastIndexOf("/") === url.length - 1)
		url = url.substring(0, url.length - 1);
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
