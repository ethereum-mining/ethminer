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
	function typeString(str)
	{
		for (var c in str)
			ts.keyClickChar(c, Qt.NoModifier, 20);
	}
}


function test_t1()
{
	waitForRendering(mainApplication.mainContent, 10000);
	mainApplication.projectModel.createProject();
	var projectDlg = mainApplication.projectModel.newProjectDialog;
	projectDlg.projectTitle = "TestProject";
	projectDlg.pathFieldText = "/tmp/MixTest"; //TODO: get platform temp path
	projectDlg.acceptAndClose();
	ts.waitForSignal(mainApplication.mainContent.codeEditor, "loadComplete()", 5000)

	ts.keyClickChar("A", Qt.ControlModifier, 20);
	ts.typeString("CCC");
	if (ts.waitForSignal(mainApplication.codeModel, "compilationComplete()", 5000))
		console.log("compiled");
	ts.waitForSignal(mainApplication.codeModel, "compilationComplete()", 5000);
}

function runTest()
{
	waitForRendering(mainApplication.mainContent, 10000);
	console.log("runtest");
	mousePress(mainApplication, 10, 10);
}

Application
{
	id: mainApplication
}
}

