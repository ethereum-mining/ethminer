import QtQuick 2.2
import QtTest 1.1
import org.ethereum.qml.TestService 1.0
import "../../qml"

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

			for (var c in str)
			{
				ts.keyPressChar(el, str[c], Qt.NoModifier, 0);
				ts.keyReleaseChar(el, str[c], Qt.NoModifier, 0);
			}
		}
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

	function clickElement(el, x, y)
	{
		ts.mouseClick(el, x, y, Qt.LeftButton, Qt.NoModifier, 10)
	}

	function test_defaultTransactionSequence()
	{
		newProject();
		editContract(
		"contract Contract {\r" +
		"	function Contract() {\r" +
		"		uint x = 69;\r" +
		"		uint y = 5;\r" +
		"		for (uint i = 0; i < y; ++i) {\r" +
		"			x += 42;\r" +
		"			z += x;\r" +
		"		}\r" +
		"	}\r" +
		"	uint z;\r" +
		"}\r"
		);
		if (!ts.waitForSignal(mainApplication.clientModel, "runComplete()", 5000))
			fail("not run");
		tryCompare(mainApplication.mainContent.rightPane.transactionLog.transactionModel, "count", 3);
	}

	function test_transactionWithParameter()
	{
		newProject();
		editContract(
		"contract Contract {\r" +
		"	function setZ(uint256 x) {\r" +
		"		z = x;\r" +
		"	}\r" +
		"	function getZ() returns(uint256) {\r" +
		"		return z;\r" +
		"	}\r" +
		"	uint z;\r" +
		"}\r"
		);
		mainApplication.projectModel.stateListModel.editState(0);
		mainApplication.projectModel.stateDialog.model.addTransaction();
		var transactionDialog = mainApplication.projectModel.stateDialog.transactionDialog;
		transactionDialog.selectFunction("setZ");
		clickElement(transactionDialog, 140, 300);
		ts.typeString("442", transactionDialog);
		transactionDialog.acceptAndClose();
		mainApplication.projectModel.stateDialog.model.addTransaction();
		transactionDialog.selectFunction("getZ");
		transactionDialog.acceptAndClose();
		mainApplication.projectModel.stateDialog.acceptAndClose();
		mainApplication.mainContent.startQuickDebugging();
		wait(1);
		if (!ts.waitForSignal(mainApplication.clientModel, "runComplete()", 5000))
			fail("not run");
		tryCompare(mainApplication.mainContent.rightPane.transactionLog.transactionModel, "count", 5);
		tryCompare(mainApplication.mainContent.rightPane.transactionLog.transactionModel.get(4), "returned", "(442)");
	}

	function test_constructorParameters()
	{
		newProject();
		editContract(
		"contract Contract {\r" +
		"	function Contract(uint256 x) {\r" +
		"		z = x;\r" +
		"	}\r" +
		"	function getZ() returns(uint256) {\r" +
		"		return z;\r" +
		"	}\r" +
		"	uint z;\r" +
		"}\r"
		);
		mainApplication.projectModel.stateListModel.editState(0);
		mainApplication.projectModel.stateDialog.model.editTransaction(2);
		var transactionDialog = mainApplication.projectModel.stateDialog.transactionDialog;
		clickElement(transactionDialog, 140, 300);
		ts.typeString("442", transactionDialog);
		transactionDialog.acceptAndClose();
		mainApplication.projectModel.stateDialog.model.addTransaction();
		transactionDialog.selectFunction("getZ");
		transactionDialog.acceptAndClose();
		mainApplication.projectModel.stateDialog.acceptAndClose();
		wait(1);
		mainApplication.mainContent.startQuickDebugging();
		if (!ts.waitForSignal(mainApplication.clientModel, "runComplete()", 5000))
			fail("not run");
		tryCompare(mainApplication.mainContent.rightPane.transactionLog.transactionModel, "count", 4);
		tryCompare(mainApplication.mainContent.rightPane.transactionLog.transactionModel.get(3), "returned", "(442)");
	}

	Application
	{
		id: mainApplication
	}
}

