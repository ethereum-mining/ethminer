/*
 * Display a row containing :
 *  - The amount of Ether.
 *  - The unit used.
 *  - User-friendly string representation of the amout of Ether (if displayFormattedValue == true).
 * 'value' has to be a QEther obj.
*/
import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Controls.Styles 1.1

RowLayout {
	id: etherEdition
	property bool displayFormattedValue;
	property bool edit;
	property variant value;
	property bool displayUnitSelection
	onValueChanged: update()
	Component.onCompleted: update()
	signal amountChanged
	signal unitChanged


	function update()
	{
		if (value)
		{
			etherValueEdit.text = value.value;
			selectUnit(value.unit);
		}
	}

	function selectUnit(unit)
	{
		units.currentIndex = unit;
	}


	DefaultTextField
	{
		onTextChanged:
		{
			if (value !== undefined)
			{
				value.setValue(text)
				formattedValue.text = value.format();
				amountChanged()
			}
		}
		readOnly: !edit
		visible: edit
		id: etherValueEdit;
	}

	ComboBox
	{
		id: units
		visible: displayUnitSelection;
		onCurrentTextChanged:
		{
			if (value)
			{
				value.setUnit(currentText);
				formattedValue.text = value.format();
				unitChanged()
			}
		}

		model: ListModel {
			id: unitsModel
			ListElement { text: "Uether"; }
			ListElement { text: "Vether"; }
			ListElement { text: "Dether"; }
			ListElement { text: "Nether"; }
			ListElement { text: "Yether"; }
			ListElement { text: "Zether"; }
			ListElement { text: "Eether"; }
			ListElement { text: "Pether"; }
			ListElement { text: "Tether"; }
			ListElement { text: "Gether"; }
			ListElement { text: "Mether"; }
			ListElement { text: "grand"; }
			ListElement { text: "ether"; }
			ListElement { text: "finney"; }
			ListElement { text: "szabo"; }
			ListElement { text: "Gwei"; }
			ListElement { text: "Mwei"; }
			ListElement { text: "Kwei"; }
			ListElement { text: "wei"; }
		}
	}

	Text
	{
		visible: displayFormattedValue
		id: formattedValue
	}
}
