import QtQuick 2.2
import QtQuick.Controls.Styles 1.1
import QtQuick.Controls 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import org.ethereum.qml.QEther 1.0
import "js/QEtherHelper.js" as QEtherHelper

Item {

	property int defaultStateIndex: -1
	property alias model: stateListModel
	property var stateList: []

	function fromPlainStateItem(s) {
		return {
			title: s.title,
			balance: QEtherHelper.createEther(s.balance.value, s.balance.unit),
			transactions: s.transactions.map(fromPlainTransactionItem)
		};
	}

	function fromPlainTransactionItem(t) {
		var r = {
			functionId: t.functionId,
			url: t.url,
			value: QEtherHelper.createEther(t.value.value, t.value.unit),
			gas: QEtherHelper.createEther(t.gas.value, t.gas.unit),
			gasPrice: QEtherHelper.createEther(t.gasPrice.value, t.gasPrice.unit),
			executeConstructor: t.executeConstructor,
			stdContract: t.stdContract,
			parameters: {}
		};
		for (var key in t.parameters) {
			var intComponent = Qt.createComponent("qrc:/qml/BigIntValue.qml");
			var param = intComponent.createObject();
			param.setValue(t.parameters[key]);
			r.parameters[key] = param;
		}
		return r;
	}

	function toPlainStateItem(s) {
		return {
			title: s.title,
			balance: { value: s.balance.value, unit: s.balance.unit },
			transactions: s.transactions.map(toPlainTransactionItem)
		};
	}

	function toPlainTransactionItem(t) {
		var r = {
			functionId: t.functionId,
			url: t.url,
			value: { value: t.value.value, unit: t.value.unit },
			gas:  { value: t.gas.value, unit: t.gas.unit },
			gasPrice: { value: t.gasPrice.value, unit: t.gasPrice.unit },
			executeConstructor: t.executeConstructor,
			stdContract: t.stdContract,
			parameters: {}
		};
		for (var key in t.parameters)
			r.parameters[key] = t.parameters[key].value();
		return r;
	}

	Connections {
		target: projectModel
		onProjectClosed: {
			stateListModel.clear();
			stateList = [];
		}
		onProjectLoaded: {
			if (!projectData.states)
				projectData.states = [];
			if (projectData.defaultStateIndex !== undefined)
				defaultStateIndex = projectData.defaultStateIndex;
			else
				defaultStateIndex = -1;
			var items = projectData.states;
			for(var i = 0; i < items.length; i++) {
				var item = fromPlainStateItem(items[i]);
				stateListModel.append(item);
				stateList.push(item);
			}
		}
		onProjectSaving: {
			projectData.states = []
			for(var i = 0; i < stateListModel.count; i++) {
				projectData.states.push(toPlainStateItem(stateList[i]));
			}
			projectData.defaultStateIndex = defaultStateIndex;
		}
		onNewProject: {
			var state = toPlainStateItem(stateListModel.createDefaultState());
			state.title = qsTr("Default");
			projectData.states = [ state ];
			projectData.defaultStateIndex = 0;
		}
	}

	StateDialog {
		id: stateDialog
		onAccepted: {
			var item = stateDialog.getItem();
			if (stateDialog.stateIndex < stateListModel.count) {
				if (stateDialog.isDefault)
					defaultStateIndex = stateIndex;
				stateList[stateDialog.stateIndex] = item;
				stateListModel.set(stateDialog.stateIndex, item);
			} else {
				if (stateDialog.isDefault)
					defaultStateIndex = 0;
				stateList.push(item);
				stateListModel.append(item);
			}

			stateListModel.save();
		}
	}

	ContractLibrary {
		id: contractLibrary;
	}

	ListModel {
		id: stateListModel

		function defaultTransactionItem() {
			return {
				value: QEtherHelper.createEther("100", QEther.Wei),
				gas: QEtherHelper.createEther("125000", QEther.Wei),
				gasPrice: QEtherHelper.createEther("10000000000000", QEther.Wei),
				executeConstructor: false,
				stdContract: false
			};
		}

		function createDefaultState() {
			var ether = QEtherHelper.createEther("100000000000000000000000000", QEther.Wei);
			var item = {
				title: "",
				balance: ether,
				transactions: []
			};

			//add all stdc contracts
			for (var i = 0; i < contractLibrary.model.count; i++) {
				var contractTransaction = defaultTransactionItem();
				var contractItem = contractLibrary.model.get(i);
				contractTransaction.url = contractItem.url;
				contractTransaction.functionId = contractItem.name;
				contractTransaction.stdContract = true;
				item.transactions.push(contractTransaction);
			};

			//add constructor
			var ctorTr = defaultTransactionItem();
			ctorTr.executeConstructor = true;
			ctorTr.functionId = qsTr("Constructor");
			item.transactions.push(ctorTr);
			return item;
		}

		function addState() {
			var item = createDefaultState();
			stateDialog.open(stateListModel.count, item, defaultStateIndex === -1);
		}

		function editState(index) {
			stateDialog.open(index, stateList[index], defaultStateIndex === index);
		}

		function debugDefaultState() {
			if (defaultStateIndex >= 0)
				runState(defaultStateIndex);
		}

		function runState(index) {
			var item = stateList[index];
			clientModel.setupState(item);
		}

		function deleteState(index) {
			stateListModel.remove(index);
			stateList.splice(index, 1);
			if (index === defaultStateIndex)
				defaultStateIndex = -1;
			save();
		}

		function save() {
			projectModel.saveProject();
		}
	}
}
