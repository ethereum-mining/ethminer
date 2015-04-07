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
		fail("Error running transaction");
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
	ts.waitForRendering(transactionDialog, 3000);
	transactionDialog.selectFunction("setZ");
	clickElement(transactionDialog, 140, 300);
	ts.typeString("442", transactionDialog);
	transactionDialog.acceptAndClose();
	mainApplication.projectModel.stateDialog.model.addTransaction();
	ts.waitForRendering(transactionDialog, 3000);
	transactionDialog.selectFunction("getZ");
	transactionDialog.acceptAndClose();
	mainApplication.projectModel.stateDialog.acceptAndClose();
	mainApplication.mainContent.startQuickDebugging();
	if (!ts.waitForSignal(mainApplication.clientModel, "runComplete()", 5000))
		fail("Error running transaction");
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
	ts.waitForRendering(transactionDialog, 3000);
	clickElement(transactionDialog, 140, 300);
	ts.typeString("442", transactionDialog);
	transactionDialog.acceptAndClose();
	mainApplication.projectModel.stateDialog.model.addTransaction();
	ts.waitForRendering(transactionDialog, 3000);
	transactionDialog.selectFunction("getZ");
	transactionDialog.acceptAndClose();
	mainApplication.projectModel.stateDialog.acceptAndClose();
	mainApplication.mainContent.startQuickDebugging();
	if (!ts.waitForSignal(mainApplication.clientModel, "runComplete()", 5000))
		fail("Error running transaction");
	tryCompare(mainApplication.mainContent.rightPane.transactionLog.transactionModel, "count", 4);
	tryCompare(mainApplication.mainContent.rightPane.transactionLog.transactionModel.get(3), "returned", "(442)");
}

function test_arrayParametersAndStorage()
{
	newProject();
	editContract(
	"	contract ArrayTest {\r" +
	"		function setM(uint256[] x) external\r" +
	"		{\r" +
	"			m = x;\r" +
	"			s = 5;\r" +
	"		}\r" +
	"		\r" +
	"		function setMV(uint72[5] x) external\r" +
	"		{\r" +
	"			mv = x;\r" +
	"			s = 42;\r" +
	"		}\r" +
	"		\r" +
	"		uint256[] m;\r" +
	"		uint72[5] mv;\r" +
	"		uint256 s;\r" +
	"	}\r");

	mainApplication.projectModel.stateListModel.editState(0);
	mainApplication.projectModel.stateDialog.model.addTransaction();
	var transactionDialog = mainApplication.projectModel.stateDialog.transactionDialog;
	ts.waitForRendering(transactionDialog, 3000);
	transactionDialog.selectFunction("setM");
	clickElement(transactionDialog, 140, 300);
	ts.typeString("4,5,6,2,10", transactionDialog);
	transactionDialog.acceptAndClose();
	mainApplication.projectModel.stateDialog.model.addTransaction();
	ts.waitForRendering(transactionDialog, 3000);
	transactionDialog.selectFunction("setMV");
	clickElement(transactionDialog, 140, 300);
	ts.typeString("13,35,1,4", transactionDialog);
	transactionDialog.acceptAndClose();
	mainApplication.projectModel.stateDialog.acceptAndClose();
	mainApplication.mainContent.startQuickDebugging();
	if (!ts.waitForSignal(mainApplication.clientModel, "debugDataReady(QObject*)", 5000))
		fail("Error running transaction");
	//debug setM
	mainApplication.clientModel.debugRecord(3);
	mainApplication.mainContent.rightPane.debugSlider.value = mainApplication.mainContent.rightPane.debugSlider.maximumValue;
	tryCompare(mainApplication.mainContent.rightPane.solStorage.item.value, "m", ["4","5","6","2","10"]);
	tryCompare(mainApplication.mainContent.rightPane.solStorage.item.value, "s", "5");
	//debug setMV
	mainApplication.clientModel.debugRecord(4);
	mainApplication.mainContent.rightPane.debugSlider.value = mainApplication.mainContent.rightPane.debugSlider.maximumValue - 1;
	tryCompare(mainApplication.mainContent.rightPane.solStorage.item.value, "mv", ["13","35","1","4","0"]);
	tryCompare(mainApplication.mainContent.rightPane.solStorage.item.value, "s", "42");
	tryCompare(mainApplication.mainContent.rightPane.solCallStack.listModel, 0, "setMV");
}
