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
	waitForExecution();
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
	waitForExecution();
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
	transactionDialog.selectFunction("getZ");
	transactionDialog.acceptAndClose();
	mainApplication.projectModel.stateDialog.acceptAndClose();
	mainApplication.mainContent.startQuickDebugging();
	waitForExecution();
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
	"			signed = 6534;\r" +
	"		}\r" +
	"		\r" +
	"		function setMV(uint72[5] x) external\r" +
	"		{\r" +
	"			mv = x;\r" +
	"			s = 42;\r" +
	"			signed = -534;\r" +
	"		}\r" +
	"		\r" +
	"		uint256[] m;\r" +
	"		uint72[5] mv;\r" +
	"		uint256 s;\r" +
	"		int48 signed;\r" +
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
	waitForExecution();
	//debug setM
	mainApplication.clientModel.debugRecord(3);
	mainApplication.mainContent.rightPane.debugSlider.value = mainApplication.mainContent.rightPane.debugSlider.maximumValue;
	tryCompare(mainApplication.mainContent.rightPane.solStorage.item.value, "m", ["4","5","6","2","10"]);
	tryCompare(mainApplication.mainContent.rightPane.solStorage.item.value, "s", "5");
	tryCompare(mainApplication.mainContent.rightPane.solStorage.item.value, "signed", "6534");
	//debug setMV
	mainApplication.clientModel.debugRecord(4);
	mainApplication.mainContent.rightPane.debugSlider.value = mainApplication.mainContent.rightPane.debugSlider.maximumValue - 1;
	tryCompare(mainApplication.mainContent.rightPane.solStorage.item.value, "mv", ["13","35","1","4","0"]);
	tryCompare(mainApplication.mainContent.rightPane.solStorage.item.value, "s", "42");
	tryCompare(mainApplication.mainContent.rightPane.solStorage.item.value, "signed", "-534");
	tryCompare(mainApplication.mainContent.rightPane.solCallStack.listModel, 0, "setMV");
}

function test_solidityDebugging()
{
	newProject();
	editContract(
	"contract Contract {\r " +
	"	function add(uint256 a, uint256 b) returns (uint256)\r " +
	"	{\r " +
	"		return a + b;\r " +
	"	}\r " +
	"	function Contract()\r " +
	"	{\r " +
	"		uint256 local = add(42, 34);\r " +
	"		storage = local;\r " +
	"	}\r " +
	"	uint256 storage;\r " +
	"}");

	mainApplication.mainContent.startQuickDebugging();
	waitForExecution();

	tryCompare(mainApplication.mainContent.rightPane.debugSlider, "maximumValue", 20);
	tryCompare(mainApplication.mainContent.rightPane.debugSlider, "value", 0);
	mainApplication.mainContent.rightPane.debugSlider.value = 13;
	tryCompare(mainApplication.mainContent.rightPane.solCallStack.listModel, 0, "add");
	tryCompare(mainApplication.mainContent.rightPane.solCallStack.listModel, 1, "Contract");
	tryCompare(mainApplication.mainContent.rightPane.solLocals.item.value, "local", "0");
	tryCompare(mainApplication.mainContent.rightPane.solStorage.item.value, "storage", undefined);
	mainApplication.mainContent.rightPane.debugSlider.value = 19;
	tryCompare(mainApplication.mainContent.rightPane.solLocals.item.value, "local", "76");
	tryCompare(mainApplication.mainContent.rightPane.solStorage.item.value, "storage", "76");
}

function test_vmDebugging()
{
	newProject();
	editContract(
	"contract Contract {\r " +
	"	function add(uint256 a, uint256 b) returns (uint256)\r " +
	"	{\r " +
	"		return a + b;\r " +
	"	}\r " +
	"	function Contract()\r " +
	"	{\r " +
	"		uint256 local = add(42, 34);\r " +
	"		storage = local;\r " +
	"	}\r " +
	"	uint256 storage;\r " +
	"}");

	mainApplication.mainContent.startQuickDebugging();
	waitForExecution();

	mainApplication.mainContent.rightPane.assemblyMode = !mainApplication.mainContent.rightPane.assemblyMode;
	tryCompare(mainApplication.mainContent.rightPane.debugSlider, "maximumValue", 41);
	tryCompare(mainApplication.mainContent.rightPane.debugSlider, "value", 0);
	mainApplication.mainContent.rightPane.debugSlider.value = 35;
	tryCompare(mainApplication.mainContent.rightPane.vmCallStack.listModel, 0, mainApplication.clientModel.contractAddresses["Contract"].substring(2));
	tryCompare(mainApplication.mainContent.rightPane.vmStorage.listModel, 0, "@ 0 (0x0)	 76 (0x4c)");
	tryCompare(mainApplication.mainContent.rightPane.vmMemory.listModel, "length", 0);
}

function test_ctrTypeAsParam()
{
	newProject();
	editContract(
	"contract C1 {\r " +
	"	function get() returns (uint256)\r " +
	"	{\r " +
	"		return 159;\r " +
	"	}\r " +
	"}\r" +
	"contract C2 {\r " +
	"   C1 c1;\r " +
	"	function getFromC1() returns (uint256)\r " +
	"	{\r " +
	"		return c1.get();\r " +
	"	}\r " +
	"   function C2(C1 _c1)\r" +
	"	{\r " +
	"       c1 = _c1;\r" +
	"	}\r " +
	"}");
	mainApplication.projectModel.stateListModel.editState(0); //C1 ctor already added
	var transactionDialog = mainApplication.projectModel.stateDialog.transactionDialog;
	mainApplication.projectModel.stateDialog.model.editTransaction(3);
	ts.waitForRendering(transactionDialog, 3000);
	clickElement(transactionDialog, 200, 300);
	ts.typeString("<C1 - 0>", transactionDialog);
	transactionDialog.acceptAndClose();
	mainApplication.projectModel.stateDialog.model.addTransaction();
	transactionDialog = mainApplication.projectModel.stateDialog.transactionDialog;
	ts.waitForRendering(transactionDialog, 3000);
	transactionDialog.selectContract("C2");
	transactionDialog.selectFunction("getFromC1");
	transactionDialog.acceptAndClose();
	mainApplication.projectModel.stateDialog.acceptAndClose();
	mainApplication.mainContent.startQuickDebugging();
	waitForExecution();

	tryCompare(mainApplication.mainContent.rightPane.transactionLog.transactionModel.get(4), "returned", "(159)");
}

