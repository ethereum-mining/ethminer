import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0
import QtQuick.Controls.Styles 1.1
import QtWebKit 3.0
import QtWebKit.experimental 1.0
import org.ethereum.qml.ProjectModel 1.0

Item {
	id: webPreview

	function reload() {



		webView.reload();
	}

	function reloadOnSave() {
		if (autoReloadOnSave.checked)
			reload();
	}

	function updateDocument(documentId, action) {
		for (var i = 0; i < pageListModel.count; i++)
			if (pageListModel.get(i).documentId === i)
				action(i);
	}

	function changePage() {
		if (pageCombo.currentIndex >=0 && pageCombo.currentIndex < pageListModel.count) {
			webView.url = pageListModel.get(pageCombo.currentIndex).path;
			reload();
		} else {
			webView.loadHtml("");
		}
	}

	Connections {
		target: ProjectModel
		onProjectSaved : reloadOnSave();
		onDocumentSaved: reloadOnSave();
		onDocumentAdded: {
			console.log("added")
			console.log(documentId)
			var document = ProjectModel.getDocument(documentId)
			if (document.isHtml)
				pageListModel.append(document);
		}
		onDocumentRemoved: {
			updateDocument(documentId, function(i) { pageListModel.remove(i) } )
		}
		onDocumentUpdated: {
			updateDocument(documentId, function(i) { pageListModel.set(i, ProjectModel.getDocument(documentId)) } )
		}

		onProjectLoaded: {
			for (var i = 0; i < target.listModel.count; i++) {
				var document = target.listModel.get(i);
				if (document.isHtml) {
					pageListModel.append(document);
					if (pageListModel.count === 1) //first page added
						changePage();
				}
			}
		}

		onProjectClosed: {
			pageListModel.clear();
		}
	}

	ListModel {
		id: pageListModel
	}

	ColumnLayout {
		anchors.fill: parent
		RowLayout {
			Layout.fillWidth: true;
			Text {
				text: qsTr("Page");
			}
			ComboBox {
				id: pageCombo
				model: pageListModel
				textRole: "name"
				currentIndex: -1
				onCurrentIndexChanged: changePage()
			}
			Button {
				text: qsTr("Reload");
				onClicked: reload()
			}
			CheckBox {
				id: autoReloadOnSave
				checked: true
				text: qsTr("Auto reload on save");
			}
		}

		ScrollView {
				Layout.fillWidth: true;
				Layout.fillHeight: true;
			WebView {
				id: webView
				url: "http://google.com"
				anchors.fill: parent
				experimental.preferences.developerExtrasEnabled: true
				experimental.itemSelector: itemSelector
			}
		}
	}

	Component {
		id: itemSelector
		MouseArea {
			// To avoid conflicting with ListView.model when inside ListView context.
			property QtObject selectorModel: model
			anchors.fill: parent
			onClicked: selectorModel.reject()
			Rectangle {
				clip: true
				width: 200
				height: Math.min(listView.contentItem.height + listView.anchors.topMargin + listView.anchors.bottomMargin
								 , Math.max(selectorModel.elementRect.y, parent.height - selectorModel.elementRect.y - selectorModel.elementRect.height))
				x: (selectorModel.elementRect.x + 200 > parent.width) ? parent.width - 200 : selectorModel.elementRect.x
				y: (selectorModel.elementRect.y + selectorModel.elementRect.height + height < parent.height ) ? selectorModel.elementRect.y + selectorModel.elementRect.height
																											  : selectorModel.elementRect.y - height;
				radius: 5
				color: "gainsboro"
				opacity: 0.8
				ListView {
					id: listView
					anchors.fill: parent
					anchors.margins: 10
					spacing: 5
					model: selectorModel.items
					delegate: Rectangle {
						color: model.selected ? "gold" : "silver"
						height: 50
						width: parent.width
						Text {
							anchors.centerIn: parent
							text: model.text
							color: model.enabled ? "black" : "gainsboro"
						}
						MouseArea {
							anchors.fill: parent
							enabled: model.enabled
							onClicked: selectorModel.accept(model.index)
						}
					}
					section.property: "group"
					section.delegate: Rectangle {
						height: 30
						width: parent.width
						color: "silver"
						Text {
							anchors.centerIn: parent
							text: section
							font.bold: true
						}
					}
				}
			}
		}

	}

}
