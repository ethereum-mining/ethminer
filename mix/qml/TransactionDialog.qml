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
	modality: Qt.WindowModal
	width: 450
	height: (paramsModel.count > 0 ? 550 : 300)
	visible: false
	color: StateDialogStyle.generic.backgroundColor
	title: qsTr("Transaction Edition")
	property int transactionIndex
	property alias transactionParams: paramsModel;
	property alias gas: gasField.value;
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
		gasField.value = item.gas;
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

	SourceSansProRegular
	{
		id: regularFont
	}

	Rectangle {
		anchors.fill: parent
		anchors.left: parent.left
		anchors.right: parent.right
		anchors.top: parent.top
		anchors.margins: 10
		color: StateDialogStyle.generic.backgroundColor

	ColumnLayout {
		id: dialogContent
		spacing: 30
		RowLayout
		{
			id: rowFunction
			Layout.fillWidth: true
			height: 150
			Label {
				Layout.preferredWidth: 75
				text: qsTr("Function")
				font.family: regularFont.name
				color: "#808080"
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
				style: ComboBoxStyle
				{
				font: regularFont.name
				}
			}
		}


		RowLayout
		{
			id: rowValue
			Layout.fillWidth: true
			height: 150
			Label {
				Layout.preferredWidth: 75
				text: qsTr("Value")
				font.family: regularFont.name
				color: "#808080"
			}
			Ether {
				id: valueField
				edit: true
				displayFormattedValue: true
			}
		}


		RowLayout
		{
			id: rowGas
			Layout.fillWidth: true
			height: 150
			Label {
				Layout.preferredWidth: 75
				text: qsTr("Gas")
				font.family: regularFont.name
				color: "#808080"
			}
			Ether {
				id: gasField
				edit: true
				displayFormattedValue: true
			}
		}

		RowLayout
		{
			id: rowGasPrice
			Layout.fillWidth: true
			height: 150
			Label {
				Layout.preferredWidth: 75
				text: qsTr("Gas Price")
				font.family: regularFont.name
				color: "#808080"
			}
			Ether {
				id: gasPriceField
				edit: true
				displayFormattedValue: true
			}
		}

		Label {
			text: qsTr("Parameters")
			Layout.preferredWidth: 75
			font.family: regularFont.name
			color: "#808080"
			visible: paramsModel.count > 0
		}

		ScrollView
		{
			Layout.fillWidth: true
			visible: paramsModel.count > 0
			ColumnLayout
			{
				id: paramRepeater
				Layout.fillWidth: true
				spacing: 10
				Repeater
				{
					anchors.fill: parent
					model: paramsModel
					visible: paramsModel.count > 0
					RowLayout
					{
						id: row
						Layout.fillWidth: true
						height: 150

						Label {
							id: typeLabel
							text: type
							font.family: regularFont.name
							Layout.preferredWidth: 50
						}

						Label {
							id: nameLabel
							text: name
							font.family: regularFont.name
							Layout.preferredWidth: 50
						}

						Label {
							id: equalLabel
							text: "="
							font.family: regularFont.name
							Layout.preferredWidth: 15
						}

						Loader
						{
							id: typeLoader
							Layout.preferredHeight: 50
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
									height: 50
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
									height: 50
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
									height: 50
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
									height: 50
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

	Component {
		id: rowDelegate
		Item {
			height: 100
		}
	}

	Component {
		id: editableDelegate
		Item {
			Loader {
				id: loaderEditor
				anchors.fill: parent
				anchors.margins: 4
				Connections {
					target: loaderEditor.item
					onTextChanged: {
						if (styleData.role === "value" && styleData.row < paramsModel.count)
							loaderEditor.updateValue(styleData.row, styleData.role, loaderEditor.item.text);
					}
				}

				function updateValue(row, role, value)
				{
					paramsModel.setProperty(styleData.row, styleData.role, value);
				}

				sourceComponent:
				{
					if (styleData.role === "value")
					{
						if (paramsModel.get(styleData.row) === undefined)
							return null;
						if (paramsModel.get(styleData.row).type.indexOf("int") !== -1)
							return intViewComp;
						else if (paramsModel.get(styleData.row).type.indexOf("bool") !== -1)
							return boolViewComp;
						else if (paramsModel.get(styleData.row).type.indexOf("string") !== -1)
							return stringViewComp;
						else if (paramsModel.get(styleData.row).type.indexOf("hash") !== -1)
							return hashViewComp;
					}
					else
						return editor;
				}



				Component {
					id: editor
					TextInput {
						id: textinput
						readOnly: true
						color: styleData.textColor
						text: styleData.value
						font.family: regularFont.name
					}
				}
			}
		}
	}
}
