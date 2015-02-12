import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0
import QtQuick.Controls.Styles 1.3
import org.ethereum.qml.QEther 1.0
import "js/TransactionHelper.js" as TransactionHelper
import "."

Window {
	id: modalTransactionDialog
	modality: Qt.ApplicationModal
	width: 450
	height: (paramsModel.count > 0 ? 500 : 300)
	visible: false
	color: StateDialogStyle.generic.backgroundColor
	title: qsTr("Edit Transaction")
	property int transactionIndex
	property alias transactionParams: paramsModel;
	property alias gas: gasValueEdit.gasValue;
	property alias gasPrice: gasPriceField.value;
	property alias transactionValue: valueField.value;
	property alias functionId: functionComboBox.currentText;
	property var itemParams;
	property bool isConstructorTransaction;
	property bool useTransactionDefaultValue: false
	property var qType;

	signal accepted;

	function open(index, item) {
		qType = [];
		rowFunction.visible = !useTransactionDefaultValue;
		rowValue.visible = !useTransactionDefaultValue;
		rowGas.visible = !useTransactionDefaultValue;
		rowGasPrice.visible = !useTransactionDefaultValue;

		transactionIndex = index;
		gasValueEdit.gasValue = item.gas;
		gasPriceField.value = item.gasPrice;
		valueField.value = item.value;
		var functionId = item.functionId;
		isConstructorTransaction = item.executeConstructor;
		rowFunction.visible = !item.executeConstructor;

		itemParams = item.parameters !== undefined ? item.parameters : {};
		functionsModel.clear();
		var functionIndex = -1;
		var functions = codeModel.code.contract.functions;
		for (var f = 0; f < functions.length; f++) {
			functionsModel.append({ text: functions[f].name });
			if (functions[f].name === item.functionId)
				functionIndex = f;
		}

		if (functionIndex == -1 && functionsModel.count > 0)
			functionIndex = 0; //@todo suggest unused function

		functionComboBox.currentIndex = functionIndex;
		paramsModel.clear();
		if (!item.executeConstructor)
			loadParameters();
		else
		{
			var parameters = codeModel.code.contract.constructor.parameters;
			for (var p = 0; p < parameters.length; p++)
				loadParameter(parameters[p]);
		}
		modalTransactionDialog.setX((Screen.width - width) / 2);
		modalTransactionDialog.setY((Screen.height - height) / 2);

		visible = true;
		valueField.focus = true;
	}

	function loadParameter(parameter)
	{
		var type = parameter.type;
		var pname = parameter.name;
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

		var param = varComponent.createObject(modalTransactionDialog);
		var value = itemParams[pname] !== undefined ? itemParams[pname] : "";

		param.setValue(value);
		param.setDeclaration(parameter);
		qType.push({ name: pname, value: param });
		paramsModel.append({ name: pname, type: type, value: value });
	}

	function loadParameters() {
		paramsModel.clear();
		if (!paramsModel)
			return;
		if (functionComboBox.currentIndex >= 0 && functionComboBox.currentIndex < functionsModel.count) {
			var func = codeModel.code.contract.functions[functionComboBox.currentIndex];
			var parameters = func.parameters;
			for (var p = 0; p < parameters.length; p++)
				loadParameter(parameters[p]);
		}
	}

	function param(name)
	{
		for (var k = 0; k < paramsModel.count; k++)
		{
			if (paramsModel.get(k).name === name)
				return paramsModel.get(k);
		}
	}

	function close()
	{
		visible = false;
	}

	function qTypeParam(name)
	{
		for (var k in qType)
		{
			if (qType[k].name === name)
				return qType[k].value;
		}
	}

	function getItem()
	{
		var item;
		if (!useTransactionDefaultValue)
		{
			item = {
				functionId: transactionDialog.functionId,
				gas: transactionDialog.gas,
				gasPrice: transactionDialog.gasPrice,
				value: transactionDialog.transactionValue,
				parameters: {},
				executeConstructor: isConstructorTransaction
			};
		}
		else
		{
			item = TransactionHelper.defaultTransaction();
			item.functionId = transactionDialog.functionId;
			item.executeConstructor = isConstructorTransaction;
		}

		if (isConstructorTransaction)
			item.functionId = qsTr("Constructor");

		var orderedQType = [];
		for (var p = 0; p < transactionDialog.transactionParams.count; p++) {
			var parameter = transactionDialog.transactionParams.get(p);
			var qtypeParam = qTypeParam(parameter.name);
			qtypeParam.setValue(parameter.value);
			orderedQType.push(qtypeParam);
			item.parameters[parameter.name] = parameter.value;
		}
		item.qType = orderedQType;
		return item;
	}

	ColumnLayout {
		anchors.fill: parent
		anchors.margins: 10

		SourceSansProLight
		{
			id: lightFont
		}

		ColumnLayout {
			id: dialogContent
			anchors.top: parent.top
			spacing: 10
			RowLayout
			{
				id: rowFunction
				Layout.fillWidth: true
				height: 150
				DefaultLabel {
					Layout.preferredWidth: 75
					text: qsTr("Function")
				}
				ComboBox {
					id: functionComboBox
					Layout.preferredWidth: 350
					currentIndex: -1
					textRole: "text"
					editable: false
					model: ListModel {
						id: functionsModel
					}
					onCurrentIndexChanged: {
						loadParameters();
					}
					style: ComboBoxStyle {
						font: lightFont.name
					}
				}
			}

			CommonSeparator
			{
				Layout.fillWidth: true
			}

			RowLayout
			{
				id: rowValue
				Layout.fillWidth: true
				height: 150
				DefaultLabel {
					Layout.preferredWidth: 75
					text: qsTr("Value")
				}
				Ether {
					id: valueField
					edit: true
					displayFormattedValue: true
				}
			}

			CommonSeparator
			{
				Layout.fillWidth: true
			}

			RowLayout
			{
				id: rowGas
				Layout.fillWidth: true
				height: 150
				DefaultLabel {
					Layout.preferredWidth: 75
					text: qsTr("Gas")
				}

				DefaultTextField
				{
					property variant gasValue
					onGasValueChanged: text = gasValue.value();
					onTextChanged: gasValue.setValue(text);
					implicitWidth: 200
					id: gasValueEdit;
				}
			}

			CommonSeparator
			{
				Layout.fillWidth: true
			}

			RowLayout
			{
				id: rowGasPrice
				Layout.fillWidth: true
				height: 150
				DefaultLabel {
					Layout.preferredWidth: 75
					text: qsTr("Gas Price")
				}
				Ether {
					id: gasPriceField
					edit: true
					displayFormattedValue: true
				}
			}

			CommonSeparator
			{
				Layout.fillWidth: true
			}

			DefaultLabel {
				id: paramLabel
				text: qsTr("Parameters:")
				Layout.preferredWidth: 75
				visible: paramsModel.count > 0
			}

			ScrollView
			{
				anchors.top: paramLabel.bottom
				anchors.topMargin: 10
				Layout.preferredWidth: 350
				Layout.fillHeight: true
				visible: paramsModel.count > 0
				Column
				{
					id: paramRepeater
					Layout.fillWidth: true
					Layout.fillHeight: true
					spacing: 3
					Repeater
					{
						height: 20 * paramsModel.count
						model: paramsModel
						visible: paramsModel.count > 0
						RowLayout
						{
							id: row
							Layout.fillWidth: true
							height: 20
							DefaultLabel {
								id: typeLabel
								text: type
								Layout.preferredWidth: 50
							}

							DefaultLabel {
								id: nameLabel
								text: name
								Layout.preferredWidth: 80
							}

							DefaultLabel {
								id: equalLabel
								text: "="
								Layout.preferredWidth: 15
							}

							Loader
							{
								id: typeLoader
								Layout.preferredWidth: 150
								function getCurrent()
								{
									return modalTransactionDialog.param(name);
								}

								Connections {
									target: typeLoader.item
									onTextChanged: {
										typeLoader.getCurrent().value = typeLoader.item.text;
									}
								}

								sourceComponent:
								{
									if (type.indexOf("int") !== -1)
										return intViewComp;
									else if (type.indexOf("bool") !== -1)
										return boolViewComp;
									else if (type.indexOf("string") !== -1)
										return stringViewComp;
									else if (type.indexOf("hash") !== -1)
										return hashViewComp;
									else
										return null;
								}

								Component
								{
									id: intViewComp
									QIntTypeView
									{
										height: 20
										width: 150
										id: intView
										text: typeLoader.getCurrent().value
									}
								}

								Component
								{
									id: boolViewComp
									QBoolTypeView
									{
										height: 20
										width: 150
										id: boolView
										defaultValue: "1"
										Component.onCompleted:
										{
											var current = typeLoader.getCurrent().value;
											(current === "" ? text = defaultValue : text = current);
										}
									}
								}

								Component
								{
									id: stringViewComp
									QStringTypeView
									{
										height: 20
										width: 150
										id: stringView
										text:
										{
											return typeLoader.getCurrent().value
										}
									}
								}

								Component
								{
									id: hashViewComp
									QHashTypeView
									{
										height: 20
										width: 150
										id: hashView
										text: typeLoader.getCurrent().value
									}
								}
							}
						}
					}
				}
			}

			CommonSeparator
			{
				Layout.fillWidth: true
				visible: paramsModel.count > 0
			}
		}

		RowLayout
		{
			anchors.bottom: parent.bottom
			anchors.right: parent.right;

			Button {
				text: qsTr("OK");
				onClicked: {
					close();
					accepted();
				}
			}
			Button {
				text: qsTr("Cancel");
				onClicked: close();
			}
		}
	}

	ListModel {
		id: paramsModel
	}
}
