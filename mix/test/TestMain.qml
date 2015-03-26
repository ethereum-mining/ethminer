import QtQuick 2.2
import QtTest 1.1
//import Qt.test.qtestroot 1.0
import "../qml"


TestCase
{

Item
{
	id: helper
	function findChild(item, childId) {
		if (item.children) {
			var searchString = "button"

			for (var idx in item.children) {
				var currentItem = item.children[idx]

				if (currentItem.id.match("^"+childId) === childId)
				  return currentItem;

				return findChild(currentItem, childId);
			}
		}
	}
}


function test_t1()
{
	waitForRendering(mainApplication.mainContent, 10000);
	mainApplication.projectModel.createProject();
	var projectDlg = helper.findChild(mainApplication, "newProjectDialog");

	if (mainApplication.appService.waitForSignal(mainApplication.codeModel, "compilationComplete()", 5000))
		console.log("compiled");
}

function runTest()
{
	waitForRendering(mainApplication.mainContent, 10000);
	console.log("runtest");
	mousePress(mainApplication, 10, 10);
}

Application {
	id: mainApplication
}

}
