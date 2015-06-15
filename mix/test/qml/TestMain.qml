import QtQuick 2.2
import QtTest 1.1
import org.ethereum.qml.TestService 1.0
import "../../qml"
import "js/TestDebugger.js" as TestDebugger
import "js/TestTutorial.js" as TestTutorial
import "js/TestMiner.js" as TestMiner
import "js/TestProject.js" as TestProject

TestCase
{
	id: tc
	TestService
	{
		id: ts
		targetWindow: mainApplication
		function typeString(str, el)
		{
			if (el === undefined)
				el = mainApplication;
			if (el.contentItem) //for dialgos
				el = el.contentItem

			for (var c in str)
			{
				ts.keyPressChar(el, str[c], Qt.NoModifier, 0);
				ts.keyReleaseChar(el, str[c], Qt.NoModifier, 0);
			}
		}
	}

	Application
	{
		id: mainApplication
		trackLastProject: false
	}

	function newProject()
	{
		mainApplication.projectModel.createProject();
		var projectDlg = mainApplication.projectModel.newProjectDialog;
		wait(30);
		projectDlg.projectTitle = "TestProject";
		projectDlg.pathFieldText = "/tmp/MixTest/" + ts.createUuid(); //TODO: get platform temp path
		projectDlg.acceptAndClose();
		wait(1);
		if (!ts.waitForSignal(mainApplication.codeModel, "compilationComplete()", 5000))
			fail("new contract not compiled");
	}

	function editContract(c)
	{
		if (mainApplication.codeModel.compiling)
			ts.waitForSignal(mainApplication.codeModel, "compilationComplete()", 5000);
		mainApplication.mainContent.codeEditor.getEditor("contract.sol").setText(c);
		if (!ts.waitForSignal(mainApplication.codeModel, "compilationComplete()", 5000))
			fail("not compiled");
		ts.keyPressChar(mainApplication, "S", Qt.ControlModifier, 200); //Ctrl+S
	}

	function waitForMining()
	{
		while (mainApplication.clientModel.mining)
			ts.waitForSignal(mainApplication.clientModel, "miningComplete()", 5000);
		wait(1); //allow events to propagate 2 times for transaction log to be updated
		wait(1);
	}

	function waitForExecution()
	{
		while (mainApplication.clientModel.running)
			ts.waitForSignal(mainApplication.clientModel, "runComplete()", 5000);
		wait(1); //allow events to propagate 2 times for transaction log to be updated
		wait(1);
	}

	function editHtml(c)
	{
		mainApplication.projectModel.openDocument("index.html");
		ts.waitForSignal(mainApplication.mainContent.codeEditor, "loadComplete()", 5000);
		mainApplication.mainContent.codeEditor.getEditor("index.html").setText(c);
		ts.keyPressChar(mainApplication, "S", Qt.ControlModifier, 200); //Ctrl+S
	}

	function createHtml(name, c)
	{
		mainApplication.projectModel.newHtmlFile();
		ts.waitForSignal(mainApplication.mainContent.codeEditor, "loadComplete()", 5000);
		var doc = mainApplication.projectModel.listModel.get(mainApplication.projectModel.listModel.count - 1);
		mainApplication.projectModel.renameDocument(doc.documentId, name);
		mainApplication.mainContent.codeEditor.getEditor(doc.documentId).setText(c);
		ts.keyPressChar(mainApplication, "S", Qt.ControlModifier, 200); //Ctrl+S
	}

	function clickElement(el, x, y)
	{
		if (el.contentItem)
			el = el.contentItem;
		ts.mouseClick(el, x, y, Qt.LeftButton, Qt.NoModifier, 10)
	}

	function test_tutorial() { TestTutorial.test_tutorial(); }
	function test_dbg_defaultTransactionSequence() { TestDebugger.test_defaultTransactionSequence(); }
	function test_dbg_transactionWithParameter() { TestDebugger.test_transactionWithParameter(); }
	function test_dbg_constructorParameters() { TestDebugger.test_constructorParameters(); }
	function test_dbg_arrayParametersAndStorage() { TestDebugger.test_arrayParametersAndStorage(); }
	function test_dbg_solidity() { TestDebugger.test_solidityDebugging(); }
	function test_dbg_vm() { TestDebugger.test_vmDebugging(); }
	function test_dbg_ctrTypeAsParam() { TestDebugger.test_ctrTypeAsParam(); }
	function test_miner_getDefaultiner() { TestMiner.test_getDefaultMiner(); }
	function test_miner_selectMiner() { TestMiner.test_selectMiner(); }
	function test_miner_mine() { TestMiner.test_mine(); }
	function test_project_contractRename() { TestProject.test_contractRename(); }
	function test_project_multipleWebPages() { TestProject.test_multipleWebPages(); }
	function test_project_multipleContractsSameFile() { TestProject.test_multipleContractsSameFile(); }
	function test_project_deleteFile() { TestProject.test_deleteFile(); }
}

