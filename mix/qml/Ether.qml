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
	onValueChanged: update()
	Component.onCompleted: update()

	function update()
	{
		if (value !== undefined)
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
		implicitWidth: 200
		onTextChanged:
		{
			if (value !== undefined)
			{
				value.setValue(text)
				formattedValue.text = value.format();
			}
		}
		readOnly: !edit
		visible: edit
		id: etherValueEdit;
	}

	SourceSansProBold {
		id: regularFont;
	}

	ComboBox
	{
		id: units
		onCurrentTextChanged:
		{
			if (value !== undefined)
			{
				value.setUnit(currentText);
				formattedValue.text = value.format();
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
		style: ComboBoxStyle {
			font: regularFont.name
		}
	}

	Text
	{
		visible: displayFormattedValue
		id: formattedValue
		font.family: regularFont.name
	}
}
