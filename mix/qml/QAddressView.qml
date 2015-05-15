import QtQuick 2.0
import QtQuick.Controls 1.3
import QtQuick.Controls.Styles 1.3

Item
{
	property alias value: textinput.text
	property alias accountRef: ctrModel
	property string subType
	property bool readOnly
	property alias currentIndex: trCombobox.currentIndex
	property alias currentText: textinput.text
	property variant accounts
	signal indexChanged()
	id: editRoot
	height: 20
	width: 320

	SourceSansProBold
	{
		id: boldFont
	}

	function currentValue() {
		return currentText;
	}

	function currentType()
	{
		return accountRef.get(trCombobox.currentIndex).type;
	}

	function current()
	{
		return accountRef.get(trCombobox.currentIndex);
	}

	function load()
	{
		accountRef.clear();
		accountRef.append({"itemid": " - "});

		if (subType === "contract" || subType === "address")
		{
			var trCr = 0;
			for (var k = 0; k < transactionsModel.count; k++)
			{
				if (k >= transactionIndex)
					break;
				var tr = transactionsModel.get(k);
				if (tr.functionId === tr.contractId /*&& (dec[1] === tr.contractId || item.subType === "address")*/)
				{
					accountRef.append({ "itemid": tr.contractId + " - " + trCr, "value": "<" + tr.contractId + " - " + trCr + ">", "type": "contract" });
					trCr++;
				}
			}
		}
		if (subType === "address")
		{
			for (k = 0; k < accounts.length; k++)
			{
				if (accounts[k].address === undefined)
					accounts[k].address = clientModel.address(accounts[k].secret);
				accountRef.append({ "itemid": accounts[k].name, "value": "0x" + accounts[k].address, "type": "address" });
			}
		}
	}

	function init()
	{
		trCombobox.visible = !readOnly
		textinput.readOnly = readOnly
		if (!readOnly)
		{
			for (var k = 0; k < ctrModel.count; k++)
			{
				if (ctrModel.get(k).value === value)
				{
					trCombobox.currentIndex = k;
					return;
				}
			}
			trCombobox.currentIndex = 0;
		}
	}

	function select(address)
	{
		for (var k = 0; k < accountRef.count; k++)
		{
			if (accountRef.get(k).value === address)
			{
				trCombobox.currentIndex = k;
				break;
			}
		}
	}

	Rectangle {
		anchors.fill: parent
		radius: 4
		anchors.verticalCenter: parent.verticalCenter
		height: 20
		TextInput {
			id: textinput
			text: value
			width: parent.width
			height: parent.width
			wrapMode: Text.WordWrap
			clip: true
			font.family: boldFont.name
			MouseArea {
				id: mouseArea
				anchors.fill: parent
				hoverEnabled: true
				onClicked: textinput.forceActiveFocus()
			}
			onTextChanged:
			{
				if (trCombobox.selected)
				{
					trCombobox.currentIndex = 0;
					trCombobox.selected = false;
				}
			}
		}
	}

	ListModel
	{
		id: ctrModel
	}

	ComboBox
	{
		property bool selected: false
		id: trCombobox
		model: ctrModel
		textRole: "itemid"
		height: 20
		anchors.verticalCenter: parent.verticalCenter
		anchors.left: textinput.parent.right
		anchors.leftMargin: 3
		onCurrentIndexChanged: {
			trCombobox.selected = false;
			if (currentText === "")
				return;
			else if (currentText !== " - ")
			{
				if (model.get(currentIndex).type === "contract")
					textinput.text = "<" + currentText + ">";
				else
					textinput.text = model.get(currentIndex).value; //address
				trCombobox.selected = true;
			}
			else if (textinput.text.indexOf("<") === 0)
			{
				textinput.text = "";
			}
			indexChanged();
		}
	}
}
