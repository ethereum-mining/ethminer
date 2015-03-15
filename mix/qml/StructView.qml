import QtQuick 2.0
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import org.ethereum.qml.QSolidityType 1.0

Item
{
	id: editRoot
	property alias membersModel: repeater.model  //js array
	property var value
	property int level: 0
	Column
	{
		id: paramRepeater
		Layout.fillWidth: true
		Layout.fillHeight: true
		spacing: 3

		Repeater
		{
			id: repeater
			height: 20 * model.length
			visible: model.length > 0
			RowLayout
			{
				id: row
				Layout.fillWidth: true
				DefaultLabel {
					id: typeLabel
					text: modelData.type.name
					Layout.preferredWidth: 50
				}

				DefaultLabel {
					id: nameLabel
					text: modelData.name
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
					sourceComponent:
					{
						var t = modelData.type.type;
						if (t === QSolidityType.SignedInteger || t === QSolidityType.UnsignedInteger)
							return Qt.createComponent("qrc:/qml/QIntTypeView.qml");
						else if (t === QSolidityType.Bool)
							return Qt.createComponent("qrc:/qml/QBoolTypeView.qml");
						else if (t === QSolidityType.Bytes)
							return Qt.createComponent("qrc:/qml/QStringTypeView.qml");
						else if (t === QSolidityType.Hash || t === QSolidityType.Address)
							return Qt.createComponent("qrc:/qml/QHashTypeView.qml");
						else if (t === QSolidityType.Struct)
							return Qt.createComponent("qrc:/qml/StructView.qml");
						else
							return undefined;
					}
					onLoaded:
					{
						var ptype = membersModel[index].type;
						var pname = membersModel[index].name;
						var vals = value;
						if (ptype.type === QSolidityType.Struct && !item.membersModel) {
							item.level = level + 1;
							item.value = getValue();
							item.membersModel = ptype.members;
						}
						else
							item.value = getValue();
						item.onValueChanged.connect(function() {
							vals[pname] = item.value;
							valueChanged();
						});

					}

					function getValue()
					{
						if (value && value[modelData.name])
							return value[modelData.name];
						else if (modelData.type.type === QSolidityType.Struct)
							return {};
						return "";
					}
				}
			}
		}
	}
}
