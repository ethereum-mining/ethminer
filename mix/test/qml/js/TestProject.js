function test_contractRename()
{
	newProject();
	waitForExecution();
	tryCompare(mainApplication.mainContent.projectNavigator.sections.itemAt(0).model.get(0), "name", "Contract");
	editContract("contract Renamed {}");
	mainApplication.mainContent.startQuickDebugging();
	waitForExecution();
	tryCompare(mainApplication.mainContent.rightPane.transactionLog.transactionModel.get(2), "contract", "Renamed");
	tryCompare(mainApplication.mainContent.projectNavigator.sections.itemAt(0).model.get(0), "name", "Renamed");
	mainApplication.projectModel.stateListModel.editState(0);
	mainApplication.projectModel.stateDialog.model.editTransaction(2);
	var transactionDialog = mainApplication.projectModel.stateDialog.transactionDialog;
	tryCompare(transactionDialog, "contractId", "Renamed");
	tryCompare(transactionDialog, "functionId", "Renamed");
	transactionDialog.close();
	mainApplication.projectModel.stateDialog.close();
}

function test_multipleWebPages()
{
	newProject();
	editHtml("<html><body><a href=\"page1.html\">page1</a></body></html>");
	createHtml("page1.html", "<html><body><div id='queryres'>Fail</div></body><script>if (web3) document.getElementById('queryres').innerText='OK'</script></html>");
	clickElement(mainApplication.mainContent.webView.webView, 1, 1);
	ts.typeString("\t\r");
	wait(300); //TODO: use a signal in qt 5.5
	mainApplication.mainContent.webView.getContent();
	ts.waitForSignal(mainApplication.mainContent.webView, "webContentReady()", 5000);
	var body = mainApplication.mainContent.webView.webContent;
	verify(body.indexOf("<div id=\"queryres\">OK</div>") != -1, "Web content not updated")
}

function test_multipleContractsSameFile()
{
	newProject();
	editContract(
	"contract C1 {}\r" +
	"contract C2 {}\r" +
	"contract C3 {}\r");
	waitForExecution();
	tryCompare(mainApplication.mainContent.rightPane.transactionLog.transactionModel, "count", 5);
	tryCompare(mainApplication.mainContent.rightPane.transactionLog.transactionModel.get(2), "contract", "C1");
	tryCompare(mainApplication.mainContent.rightPane.transactionLog.transactionModel.get(3), "contract", "C2");
	tryCompare(mainApplication.mainContent.rightPane.transactionLog.transactionModel.get(4), "contract", "C3");
}

function test_deleteFile()
{
	newProject();
	var path = mainApplication.projectModel.projectPath;
	createHtml("page1.html", "<html><body><div id='queryres'>Fail</div></body><script>if (web3) document.getElementById('queryres').innerText='OK'</script></html>");
	createHtml("page2.html", "<html><body><div id='queryres'>Fail</div></body><script>if (web3) document.getElementById('queryres').innerText='OK'</script></html>");
	createHtml("page3.html", "<html><body><div id='queryres'>Fail</div></body><script>if (web3) document.getElementById('queryres').innerText='OK'</script></html>");
	mainApplication.projectModel.removeDocument("page2.html");
	mainApplication.projectModel.closeProject(function(){});
	mainApplication.projectModel.loadProject(path);
	var doc = mainApplication.projectModel.getDocument("page2.html");
	verify(!doc, "page2.html has not been removed");
}
