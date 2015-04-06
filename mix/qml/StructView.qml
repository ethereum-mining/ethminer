import QtQuick 2.0
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import org.ethereum.qml.QSolidityType 1.0

Column
{
	id: root
	property alias members: repeater.model  //js array
	property var value: ({})
	Layout.fillWidth: true

	Repeater
	{
		id: repeater
		visible: model.length > 0
		Layout.fillWidth: true
		RowLayout
		{
			id: row
			height: 20 + (members[index].type.category === QSolidityType.Struct ? (20 * members[index].type.members.length) : 0)
			Layout.fillWidth: true
			DefaultLabel {
				height: 20
				id: typeLabel
				text: modelData.type.name
				anchors.verticalCenter: parent.verticalCenter
			}

			DefaultLabel {
				id: nameLabel
				text: modelData.name
				anchors.verticalCenter: parent.verticalCenter
			}

			DefaultLabel {
				id: equalLabel
				text: "="
				anchors.verticalCenter: parent.verticalCenter
			}
			Loader
			{
				id: typeLoader
				anchors.verticalCenter: parent.verticalCenter
				sourceComponent:
				{
					var t = modelData.type.category;
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
					var ptype = members[index].type;
					var pname = members[index].name;
					var vals = value;
					if (ptype.category === QSolidityType.Struct && !item.members)
					{
						item.value = getValue();
						item.members = ptype.members;
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
					var r = "";
					if (value && value[modelData.name] !== undefined)
						r = value[modelData.name];
					else if (modelData.type.category === QSolidityType.Struct)
						r = {};
					if (Array.isArray(r))
						r = r.join(", ");
					return r;
				}
			}
		}
	}
}
