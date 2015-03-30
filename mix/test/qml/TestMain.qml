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
		function typeString(str)
		{
			for (var c in str)
			{
				ts.keyPressChar(str[c], Qt.NoModifier, 0);
				ts.keyReleaseChar(str[c], Qt.NoModifier, 0);
			}
		}
	}

	function newProject()
	{
		waitForRendering(mainApplication.mainContent, 10000);
		mainApplication.projectModel.createProject();
		var projectDlg = mainApplication.projectModel.newProjectDialog;
		projectDlg.projectTitle = "TestProject";
		projectDlg.pathFieldText = "/tmp/MixTest"; //TODO: get platform temp path
		projectDlg.acceptAndClose();
		ts.waitForSignal(mainApplication.mainContent.codeEditor, "loadComplete()", 5000)
		wait(300);
	}

	function editContract(c)
	{
		mainApplication.mainContent.codeEditor.getEditor("contract.sol").setText(c);
		ts.keyPressChar("S", Qt.ControlModifier, 200); //Ctrl+S
		if (!ts.waitForSignal(mainApplication.codeModel, "compilationComplete()", 5000))
			fail("not compiled");
	}

	function clickElemet(el)
	{
		mouseClick(el, 0, 0, Qt.LeftButton, Qt.NoModifier, 10)
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
		tryCompare(mainApplication.mainContent.rightPane.transactionLog.transactionModel, "count", 3);
	}

	function test_transactionWithParameter()
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
		//TODO: edit transaction params and check results
	}

	function test_constructorParameters()
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
		transactionDialog.functionId = "setZ";


		//TODO: edit transaction params and check results
	}

	Application
	{
		id: mainApplication
	}
}

