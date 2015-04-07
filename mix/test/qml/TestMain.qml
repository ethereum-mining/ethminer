import QtQuick 2.2
import QtTest 1.1
import org.ethereum.qml.TestService 1.0
import "../../qml"
import "js/TestDebugger.js" as TestDebugger
import "js/TestTutorial.js" as TestTutorial

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
	}

	function newProject()
	{
		waitForRendering(mainApplication.mainContent, 10000);
		mainApplication.projectModel.createProject();
		var projectDlg = mainApplication.projectModel.newProjectDialog;
		wait(30);
		projectDlg.projectTitle = "TestProject";
		projectDlg.pathFieldText = "/tmp/MixTest"; //TODO: get platform temp path
		projectDlg.acceptAndClose();
		wait(30);
	}

	function editContract(c)
	{
		mainApplication.mainContent.codeEditor.getEditor("contract.sol").setText(c);
		ts.keyPressChar(mainApplication, "S", Qt.ControlModifier, 200); //Ctrl+S
		if (!ts.waitForSignal(mainApplication.codeModel, "compilationComplete()", 5000))
			fail("not compiled");
	}

	function editHtml(c)
	{
		mainApplication.projectModel.openDocument("index.html");
		wait(1);
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
}

