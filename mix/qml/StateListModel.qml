import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Window 2.2
import QtQuick.Layouts 1.1
import org.ethereum.qml.QEther 1.0
import "js/QEtherHelper.js" as QEtherHelper
import "js/TransactionHelper.js" as TransactionHelper

Item {

	property alias model: stateListModel
	property var stateList: []
	property alias stateDialog: stateDialog
	property string defaultAccount: "cb73d9408c4720e230387d956eb0f829d8a4dd2c1055f96257167e14e7169074" //support for old project

	function fromPlainStateItem(s) {
		if (!s.accounts)
			s.accounts = [stateListModel.newAccount("1000000", QEther.Ether, defaultAccount)]; //support for old project
		if (!s.contracts)
			s.contracts = [];

		var ret = {};
		ret.title = s.title;
		ret.transactions = s.transactions.filter(function(t) { return !t.stdContract; }).map(fromPlainTransactionItem); //support old projects by filtering std contracts
		if (s.blocks)
			ret.blocks = s.blocks.map(fromPlainBlockItem);
		ret.accounts = s.accounts.map(fromPlainAccountItem);
		ret.contracts = s.contracts.map(fromPlainAccountItem);
		ret.miner = s.miner;

		// support old projects
		if (!ret.blocks)
		{
			ret.blocks = [{
							  hash: "",
							  number: -1,
							  transactions: [],
							  status: "pending"
						  }]
			for (var j in ret.transactions)
				ret.blocks[0].transactions.push(fromPlainTransactionItem(toPlainTransactionItem(ret.transactions[j])))
		}
		return ret;
	}

	function fromPlainAccountItem(t)
	{
		return {
			name: t.name,
			address: t.address,
			secret: t.secret,
			balance: QEtherHelper.createEther(t.balance.value, t.balance.unit),
			storage: t.storage,
			code: t.code,
		};
	}

	function fromPlainTransactionItem(t) {
		if (!t.sender)
			t.sender = defaultAccount; //support for old project

		var r = {
			type: t.type,
			contractId: t.contractId,
			functionId: t.functionId,
			url: t.url,
			value: QEtherHelper.createEther(t.value.value, t.value.unit),
			gas: QEtherHelper.createBigInt(t.gas.value),
			gasPrice: QEtherHelper.createEther(t.gasPrice.value, t.gasPrice.unit),
			gasAuto: t.gasAuto,
			parameters: {},
			sender: t.sender,
			isContractCreation: t.isContractCreation,
			label: t.label,
			isFunctionCall: t.isFunctionCall,
			saveStatus: t.saveStatus
		};

		if (r.saveStatus === undefined)
			r.saveStatus = true

		if (r.isFunctionCall === undefined)
			r.isFunctionCall = true;

		if (!r.label)
			r.label = r.contractId + " - " + r.functionId;

		if (r.isContractCreation === undefined)
			r.isContractCreation = r.functionId === r.contractId;

		for (var key in t.parameters)
			r.parameters[key] = t.parameters[key];

		return r;
	}

	function fromPlainBlockItem(b)
	{
		var r = {
			hash: b.hash,
			number: b.number,
			transactions: b.transactions.filter(function(t) { return !t.stdContract; }).map(fromPlainTransactionItem), //support old projects by filtering std contracts
			status: b.status
		}
		return r;
	}

	function toPlainStateItem(s) {
		return {
			title: s.title,
			blocks: s.blocks.map(toPlainBlockItem),
			transactions: s.transactions.map(toPlainTransactionItem),
			accounts: s.accounts.map(toPlainAccountItem),
			contracts: s.contracts.map(toPlainAccountItem),
			miner: s.miner
		};
	}

	function getParamType(param, params)
	{
		for (var k in params)
		{
			if (params[k].declaration.name === param)
				return params[k].declaration.type;
		}
		return '';
	}

	function toPlainBlockItem(b)
	{
		var r = {
			hash: b.hash,
			number: b.number,
			transactions: b.transactions.map(toPlainTransactionItem),
			status: b.status
		}
		return r;
	}

	function toPlainAccountItem(t)
	{
		return {
			name: t.name,
			secret: t.secret,
			balance: {
				value: t.balance.value,
				unit: t.balance.unit
			},
			address: t.address,
			storage: t.storage,
			code: t.code,
		};
	}

	function toPlainTransactionItem(t) {
		var r = {
			type: t.type,
			contractId: t.contractId,
			functionId: t.functionId,
			url: t.url,
			value: { value: t.value.value, unit: t.value.unit },
			gas: { value: t.gas.value() },
			gasAuto: t.gasAuto,
			gasPrice: { value: t.gasPrice.value, unit: t.gasPrice.unit },
			sender: t.sender,
			parameters: {},
			isContractCreation: t.isContractCreation,
			label: t.label,
			isFunctionCall: t.isFunctionCall,
			saveStatus: t.saveStatus
		};
		for (var key in t.parameters)
			r.parameters[key] = t.parameters[key];
		return r;
	}

	Connections {
		target: projectModel
		onProjectClosed: {
			stateListModel.clear();
			stateList = [];
			codeModel.reset();
		}
		onProjectLoading: stateListModel.loadStatesFromProject(projectData);
		onProjectFileSaving: {
			projectData.states = []
			for(var i = 0; i < stateListModel.count; i++)
			{
				projectData.states.push(toPlainStateItem(stateList[i]));
				stateListModel.set(i, stateList[i]);
			}
			projectData.defaultStateIndex = stateListModel.defaultStateIndex;
			stateListModel.data = projectData
		}
		onNewProject: {
			var state = toPlainStateItem(stateListModel.createDefaultState());
			state.title = qsTr("Default");
			projectData.states = [ state ];
			projectData.defaultStateIndex = 0;
			stateListModel.loadStatesFromProject(projectData);
		}
	}

	Connections {
		target: codeModel
		onNewContractCompiled: {
			stateListModel.addNewContracts();
		}
		onContractRenamed: {
			stateListModel.renameContracts(_oldName, _newName);
		}
	}

	StateDialog {
		id: stateDialog
		onAccepted: {
			var item = stateDialog.getItem();
			saveState(item);
		}

		function saveState(item)
		{
			stateList[stateDialog.stateIndex].accounts = item.accounts
			stateList[stateDialog.stateIndex].contracts = item.contracts
			stateListModel.get(stateDialog.stateIndex).accounts = item.accounts
			stateListModel.get(stateDialog.stateIndex).contracts = item.contracts
			stateListModel.accountsValidated(item.accounts)
			stateListModel.contractsValidated(item.contracts)
			stateListModel.get(stateDialog.stateIndex).miner = item.miner
			stateList[stateDialog.stateIndex].miner = item.miner
			if (item.defaultState)
			{
				stateListModel.defaultStateIndex = stateDialog.stateIndex
				stateListModel.defaultStateChanged()
			}
		}
	}

	ListModel {
		id: stateListModel
		property int defaultStateIndex: 0
		property variant data
		signal accountsValidated(var _accounts)
		signal contractsValidated(var _contracts)
		signal defaultStateChanged;
		signal stateListModelReady;
		signal stateRun(int index)
		signal stateDeleted(int index)

		function defaultTransactionItem()
		{
			return TransactionHelper.defaultTransaction();
		}

		function newAccount(_balance, _unit, _secret)
		{
			if (!_secret)
				_secret = clientModel.newSecret();
			var address = clientModel.address(_secret);
			var name = qsTr("Account") + "-" + address.substring(0, 4);
			var amount = QEtherHelper.createEther(_balance, _unit)			
			return { name: name, secret: _secret, balance: amount, address: address };
		}

		function duplicateState(index)
		{
			var state = stateList[index]
			var item = fromPlainStateItem(toPlainStateItem(state))
			item.title = qsTr("Copy of") + " " + state.title
			appendState(item)
			save()
		}

		function createEmptyBlock()
		{
			return {
				hash: "",
				number: -1,
				transactions: [],
				status: "pending"
			}
		}

		function createDefaultState() {
			var item = {
				title: "",
				transactions: [],
				accounts: [],
				contracts: [],
				blocks: [{ status: "pending", number: -1, hash: "", transactions: []}]
			};

			var account = newAccount("1000000", QEther.Ether, defaultAccount)
			item.accounts.push(account);
			item.miner = account;

			//add constructors, //TODO: order by dependencies
			for(var c in codeModel.contracts) {
				var ctorTr = defaultTransactionItem();
				ctorTr.functionId = c;
				ctorTr.contractId = c;
				ctorTr.label = ctorTr.contractId + "." + ctorTr.contractId + "()"
				ctorTr.sender = item.accounts[0].secret;
				item.transactions.push(ctorTr);
				item.blocks[0].transactions.push(ctorTr)
			}
			return item;
		}

		function renameContracts(oldName, newName) {
			var changed = false;
			for(var c in codeModel.contracts) {
				for (var s = 0; s < stateListModel.count; s++) {
					var state = stateList[s];
					for (var t = 0; t < state.transactions.length; t++) {
						var transaction = state.transactions[t];
						if (transaction.contractId === oldName) {
							transaction.contractId = newName;
							if (transaction.functionId === oldName)
								transaction.functionId = newName;
							changed = true;
							state.transactions[t] = transaction;
						}
					}
					stateListModel.set(s, state);
					stateList[s] = state;
				}
			}
			if (changed)
				save();
		}

		function addNewContracts() {
			//add new contracts to empty states
			var changed = false;
			for (var c in codeModel.contracts) {
				for (var s = 0; s < stateListModel.count; s++) {
					var state = stateList[s];
					if (state.transactions.length === 0) {
						//append this contract
						var ctorTr = defaultTransactionItem();
						ctorTr.functionId = c;
						ctorTr.contractId = c;
						ctorTr.label = ctorTr.contractId + "." + ctorTr.contractId + "()";
						ctorTr.sender = state.accounts[0].secret;
						state.transactions.push(ctorTr);
						changed = true;
						stateListModel.set(s, state);
						stateList[s] = state;
					}
				}
			}
			if (changed)
				save();
		}

		function addState() {
			var item = createDefaultState();
			stateDialog.open(stateListModel.count, item, false);
		}

		function appendState(item)
		{
			stateListModel.append(item);
			stateList.push(item);
		}

		function editState(index) {
			stateDialog.open(index, stateList[index], defaultStateIndex === index);
		}

		function getState(index) {
			return stateList[index];
		}

		function debugDefaultState() {
			if (defaultStateIndex >= 0 && defaultStateIndex < stateList.length)
				runState(defaultStateIndex);
		}

		function runState(index) {
			var item = stateList[index];
			clientModel.setupScenario(item);
			stateRun(index);
		}

		function deleteState(index) {
			stateListModel.remove(index);
			stateList.splice(index, 1);
			if (index === defaultStateIndex)
			{
				defaultStateIndex = 0;
				defaultStateChanged();
			}
			else if (defaultStateIndex > index)
				defaultStateIndex--;
			save();
			stateDeleted(index);
		}

		function save() {
			projectModel.saveProject();
		}

		function defaultStateName()
		{
			if (stateList.length > 0)
				return stateList[defaultStateIndex].title;
			else
				return ""
		}

		function reloadStateFromProject(index)
		{
			if (data)
			{
				var item = fromPlainStateItem(data.states[index])
				stateListModel.set(index, item)
				stateList[index] = item
				return item
			}
		}

		function loadStatesFromProject(projectData)
		{
			data = projectData
			if (!projectData.states)
				projectData.states = [];
			if (projectData.defaultStateIndex !== undefined)
				defaultStateIndex = projectData.defaultStateIndex;
			else
				defaultStateIndex = 0;
			var items = projectData.states;
			stateListModel.clear();
			stateList = [];
			for(var i = 0; i < items.length; i++) {
				var item = fromPlainStateItem(items[i]);
				stateListModel.append(item);
				stateList.push(item);
			}
			stateListModelReady();
		}
	}
}
