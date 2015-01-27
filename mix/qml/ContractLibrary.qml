import QtQuick 2.2

Item {
	id: contractLibrary
	property alias model: contractListModel;

	Connections {
		target: appContext
		Component.onCompleted: {

			//TODO: load a list, dependencies, ets, from external files
			var configSource = fileIo.readFile("qrc:///stdc/config.sol");
			var nameRegSource = fileIo.readFile("qrc:///stdc/namereg.sol");
			contractListModel.append({
				name: "Config",
				url: "qrc:///stdc/config.sol",
				source: configSource
			});
			contractListModel.append({
				name: "NameReg",
				url: "qrc:///stdc/namereg.sol",
				source: nameRegSource
			});
		}
	}

	ListModel {
		id: contractListModel
	}
}

