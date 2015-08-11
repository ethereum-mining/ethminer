import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Layouts 1.1

DebugInfoList
{
	id: storage
	collapsible: true
	title : qsTr("Storage")
	componentDelegate: structComp

	Component
	{
		id: structComp
		ScrollView
		{
			property alias members: typeLoader.members;
			property alias value: typeLoader.value;
			anchors.fill: parent
			anchors.leftMargin: 10
			StructView
			{
				id: typeLoader
				members: []
				value: {}
				context: "variable"
				width: parent.width
			}
		}
	}

	function setData(members, values)  {
		storage.item.value = {};
		storage.item.members = [];
		storage.item.value = values; //TODO: use a signal for this?
		storage.item.members = members;
	}
}

