import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import Qt.labs.settings 1.0
import org.ethereum.qml.QEther 1.0
import "js/Debugger.js" as Debugger
import "js/ErrorLocationFormater.js" as ErrorLocationFormater
import "js/TransactionHelper.js" as TransactionHelper
import "js/QEtherHelper.js" as QEtherHelper
import "."

ColumnLayout {
	id: root
	property alias title: titleLabel.text
	property variant data

	function key(index)
	{
	}

	function value(index)
	{
	}

	RowLayout
	{
		Label
		{
			id: titleLabel
		}
	}

	RowLayout
	{
		Repeater
		{
			id: repeaterKeyValue
			RowLayout
			{
				Rectangle
				{
					id: key
					Label
					{
						text: {
							return root.key(index)
						}
					}
				}
				Rectangle
				{
					id: value
					Label
					{
						text: {
							return root.value(index)
						}
					}
				}
			}
		}
	}
}

