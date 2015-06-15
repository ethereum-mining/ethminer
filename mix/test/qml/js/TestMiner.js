function test_getDefaultMiner()
{
	newProject();
	var state = mainApplication.projectModel.stateListModel.get(0);
	compare(state.miner.secret, "cb73d9408c4720e230387d956eb0f829d8a4dd2c1055f96257167e14e7169074");
}

function test_selectMiner()
{
	newProject();
	mainApplication.projectModel.stateListModel.editState(0);
	var account = mainApplication.projectModel.stateDialog.newAccAction.add();
	account = mainApplication.projectModel.stateDialog.newAccAction.add();
	mainApplication.projectModel.stateDialog.minerComboBox.currentIndex = 2;
	ts.waitForRendering(mainApplication.projectModel.stateDialog.minerComboBox, 3000);
	mainApplication.projectModel.stateDialog.acceptAndClose();
	var state = mainApplication.projectModel.stateListModel.get(0);
	compare(state.miner.secret, account.secret);
}

function test_mine()
{
	newProject();
	mainApplication.mainContent.startQuickDebugging();
	waitForExecution();
	mainApplication.clientModel.mine();
	waitForMining();
	wait(1000); //there need to be at least 1 sec diff between block times
	mainApplication.clientModel.mine();
	waitForMining();
	tryCompare(mainApplication.mainContent.rightPane.transactionLog.transactionModel.get(3), "contract", " - Block - ");
	tryCompare(mainApplication.mainContent.rightPane.transactionLog.transactionModel.get(4), "contract", " - Block - ");
}

