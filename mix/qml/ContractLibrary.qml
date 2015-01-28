import QtQuick 2.2

Item {
	id: contractLibrary
	property alias model: contractListModel;

	Connections {
		target: appContext
		Component.onCompleted: {

			//TODO: load a list, dependencies, ets, from external files
			contractListModel.append({
				name: "Config",
				url: "qrc:///stdc/std.sol",
			});
			contractListModel.append({
				name: "NameReg",
				url: "qrc:///stdc/std.sol",
			});
		}
	}

	ListModel {
		id: contractListModel
	}
}

