import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Window 2.2
import QtQuick.Layouts 1.1
import org.ethereum.qml.QEther 1.0
import "js/QEtherHelper.js" as QEtherHelper

Item {

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
			gas: QEtherHelper.createBigInt(t.gas.value), //t.gas,//QEtherHelper.createEther(t.gas.value, t.gas.unit),
			gasPrice: QEtherHelper.createEther(t.gasPrice.value, t.gasPrice.unit),
			executeConstructor: t.executeConstructor,
			stdContract: t.stdContract,
			parameters: {}
		};
		var qType = [];
		for (var key in t.parameters)
		{
			r.parameters[key] = t.parameters[key].value;
			var type = t.parameters[key].type;
			var varComponent;
			if (type.indexOf("int") !== -1)
				varComponent = Qt.createComponent("qrc:/qml/QIntType.qml");
			else if (type.indexOf("real") !== -1)
				varComponent = Qt.createComponent("qrc:/qml/QRealType.qml");
			else if (type.indexOf("string") !== -1 || type.indexOf("text") !== -1)
				varComponent = Qt.createComponent("qrc:/qml/QStringType.qml");
			else if (type.indexOf("hash") !== -1 || type.indexOf("address") !== -1)
				varComponent = Qt.createComponent("qrc:/qml/QHashType.qml");
			else if (type.indexOf("bool") !== -1)
				varComponent = Qt.createComponent("qrc:/qml/QBoolType.qml");

			var param = varComponent.createObject(stateListModel);
			var dec = Qt.createComponent("qrc:/qml/QVariableDeclaration.qml");
			param.setDeclaration(dec.createObject(stateListModel, { "type": type }));
			param.setValue(r.parameters[key]);
			qType.push(param);
		}
		r.qType = qType;
		return r;
	}

	function toPlainStateItem(s) {
		return {
			title: s.title,
			balance: { value: s.balance.value, unit: s.balance.unit },
			transactions: s.transactions.map(toPlainTransactionItem)
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

	function toPlainTransactionItem(t) {
		var r = {
			functionId: t.functionId,
			url: t.url,
			value: { value: t.value.value, unit: t.value.unit },
			gas: { value: t.gas.value() },
			gasPrice: { value: t.gasPrice.value, unit: t.gasPrice.unit },
			executeConstructor: t.executeConstructor,
			stdContract: t.stdContract,
			parameters: {}
		};
		for (var key in t.parameters)
		{
			var param = {
				name: key,
				value: t.parameters[key],
				type: getParamType(key, t.qType)
			}
			r.parameters[key] = param;
		}
		return r;
	}

	Connections {
		target: projectModel
		onProjectClosed: {
			stateListModel.clear();
			stateList = [];
		}
		onProjectLoading: stateListModel.loadStatesFromProject(projectData);
		onProjectSaving: {
			projectData.states = []
			for(var i = 0; i < stateListModel.count; i++) {
				projectData.states.push(toPlainStateItem(stateList[i]));
			}
			projectData.defaultStateIndex = stateListModel.defaultStateIndex;
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
					stateListModel.defaultStateIndex = stateIndex;
				stateList[stateDialog.stateIndex] = item;
				stateListModel.set(stateDialog.stateIndex, item);
			} else {
				if (stateDialog.isDefault)
					stateListModel.defaultStateIndex = 0;
				stateList.push(item);
				stateListModel.append(item);
			}
			if (stateDialog.isDefault)
				stateListModel.defaultStateChanged();
			stateListModel.save();
		}
	}

	ContractLibrary {
		id: contractLibrary;
	}

	ListModel {
		id: stateListModel

		property int defaultStateIndex: 0
		signal defaultStateChanged;
		signal stateListModelReady;
		signal stateRun(int index)

		function defaultTransactionItem() {
			return {
				value: QEtherHelper.createEther("100", QEther.Wei),
				gas: QEtherHelper.createBigInt("125000"),
				gasPrice: QEtherHelper.createEther("10000000000000", QEther.Wei),
				executeConstructor: false,
				stdContract: false
			};
		}

		function createDefaultState() {
			//var ether = QEtherHelper.createEther("100000000000000000000000000", QEther.Wei);
			var ether = QEtherHelper.createEther("1000000", QEther.Ether);
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
			stateDialog.open(stateListModel.count, item, false);
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
			save();
		}

		function save() {
			projectModel.saveProject();
		}

		function defaultStateName()
		{
			return stateList[defaultStateIndex].title;
		}

		function loadStatesFromProject(projectData)
		{
			if (!projectData.states)
				projectData.states = [];
			if (projectData.defaultStateIndex !== undefined)
				defaultStateIndex = projectData.defaultStateIndex;
			else
				defaultStateIndex = 0;
			var items = projectData.states;
			for(var i = 0; i < items.length; i++) {
				var item = fromPlainStateItem(items[i]);
				stateListModel.append(item);
				stateList.push(item);
			}
			stateListModelReady();
		}
	}
}
