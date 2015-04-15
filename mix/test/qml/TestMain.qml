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
		projectDlg.pathFieldText = "/tmp/MixTest"; //TODO: get platform temp path
		projectDlg.acceptAndClose();
		wait(1);
		if (!ts.waitForSignal(mainApplication.codeModel, "compilationComplete()", 5000))
			fail("new contract not compiled");
	}

	function editContract(c)
	{
		mainApplication.mainContent.codeEditor.getEditor("contract.sol").setText(c);
		if (!ts.waitForSignal(mainApplication.codeModel, "compilationComplete()", 5000))
			fail("not compiled");
		ts.keyPressChar(mainApplication, "S", Qt.ControlModifier, 200); //Ctrl+S
	}

	function editHtml(c)
	{
		mainApplication.projectModel.openDocument("index.html");
		ts.waitForSignal(mainApplication.mainContent.codeEditor, "loadComplete()", 5000);
		mainApplication.mainContent.codeEditor.getEditor("index.html").setText(c);
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
	function test_miner_getDefaultiner() { TestMiner.test_getDefaultMiner(); }
	function test_miner_selectMiner() { TestMiner.test_selectMiner(); }
	function test_project_contractRename() { TestProject.test_contractRename(); }
}

