import QtQuick 2.2
import QtQuick.Controls.Styles 1.2
import QtQuick.Controls 1.2
import QtQuick.Dialogs 1.2
import QtQuick.Layouts 1.1

Rectangle {
	color: "transparent"
	id: transactionListContainer
	focus: true
	anchors.topMargin: 10
	anchors.left: parent.left
	height: parent.height
	width: parent.width

	Connections {
		target: appContext
		onProjectLoaded: {
			var items = JSON.parse(_json);
			for(var i = 0; i < items.length; i++) {
				transactionListModel.append(items[i]);
			}
		}
	}

	ListView {
		anchors.top: parent.top
		height: parent.height
		width: parent.width
		id: transactionList
		model: ListModel {
			id: transactionListModel

			function runTransaction(index) {
				var item = transactionListModel.get(index);
				debugModel.debugTransaction(item);
			}
		}

		delegate: renderDelegate
	}

	Button {
		anchors.bottom: parent.bottom
		text: qsTr("Add")
		onClicked:
		{
			// Set next id here to work around Qt bug
			// https://bugreports.qt-project.org/browse/QTBUG-41327
			// Second call to signal handler would just edit the item that was just created, no harm done
			var item = {
				title: "",
				value: "",
				functionId: "",
				gas: "1000000000000",
				gasPrice: "100000"
			};

			transactionDialog.reset(transactionListModel.count, item);
			transactionDialog.open();
			transactionDialog.focus = true;
		}
	}

	TransactionDialog {
		id: transactionDialog
		onAccepted: {
			var item = {
				title: transactionDialog.transactionTitle,
				functionId: transactionDialog.functionId,
				gas: transactionDialog.gas,
				gasPrice: transactionDialog.gasPrice,
				value: transactionDialog.transactionValue,
				parameters: {}
			}
			for (var p = 0; p < transactionDialog.transactionParams.count; p++) {
				var parameter = transactionDialog.transactionParams.get(p);
				item.parameters[parameter.name] = parameter.value;
			}

			console.log(item.title);
			if (transactionDialog.transactionIndex < transactionListModel.count)
				transactionListModel.set(transactionDialog.transactionIndex, item);
			else
				transactionListModel.append(item);

			var items = [];
			for (var i = 0; i < transactionListModel.count; i++)
				items.push(transactionListModel.get(i));
			var json = JSON.stringify(items, function(key, value) { return key === "objectName" ? undefined : value; });
			appContext.saveProject(json);
		}
	}

	Component {
		id: renderDelegate
		Item {
			id: wrapperItem
			height: 20
			width: parent.width
			RowLayout
			{
				anchors.fill: parent
				Text {
					//anchors.fill: parent
					Layout.fillWidth: true
					Layout.fillHeight: true
					text: title
					font.pointSize: 12
					verticalAlignment: Text.AlignBottom
				}
				ToolButton {
					text: qsTr("Edit");
					Layout.fillHeight: true
					onClicked: {
						transactionDialog.reset(index, transactionListModel.get(index));
						transactionDialog.open();
						transactionDialog.focus = true;
					}
				}
				ToolButton {
					text: qsTr("Delete");
					Layout.fillHeight: true
					onClicked: {
					}
				}
				ToolButton {
					text: qsTr("Run");
					Layout.fillHeight: true
					onClicked: {
						transactionListModel.runTransaction(index);
					}
				}
			}
		}
	}
}

