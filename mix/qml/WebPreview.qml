import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0
import QtQuick.Controls.Styles 1.1
import QtWebEngine 1.0
import Qt.WebSockets 1.0
import QtWebEngine.experimental 1.0

Item {
	id: webPreview
	property string pendingPageUrl: ""
	property bool initialized: false

	function setPreviewUrl(url) {
		if (!initialized)
			pendingPageUrl = url;
		else {
			pendingPageUrl = "";
			webView.runJavaScript("loadPage(\"" + url + "\")");
		}
	}

	function reload() {
		webView.runJavaScript("reloadPage()");
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
			setPreviewUrl(pageListModel.get(pageCombo.currentIndex).path);
		} else {
			setPreviewUrl("");
		}
	}
	Connections {
		target: appContext
		onAppLoaded: {
			//We need to load the container using file scheme so that web security would allow loading local files in iframe
			var containerPage = fileIo.readFile("qrc:///qml/html/WebContainer.html");
			webView.loadHtml(containerPage, "file:///")

		}
	}

	Connections {
		target: projectModel
		onProjectSaved : reloadOnSave();
		onDocumentSaved: reloadOnSave();
		onDocumentAdded: {
			console.log("added")
			console.log(documentId)
			var document = projectModel.getDocument(documentId)
			if (document.isHtml)
				pageListModel.append(document);
		}
		onDocumentRemoved: {
			updateDocument(documentId, function(i) { pageListModel.remove(i) } )
		}
		onDocumentUpdated: {
			updateDocument(documentId, function(i) { pageListModel.set(i, projectModel.getDocument(documentId)) } )
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

	WebSocketServer {
		id: socketServer
		listen: true
		name: "mix"
		onClientConnected:
		{
			webSocket.onTextMessageReceived.connect(function(message) {
				console.log("rpc: " + message);
			});
		}
	}

	ColumnLayout {
		anchors.fill: parent

		RowLayout {
			anchors.top: parent.top
			Layout.fillWidth: true;
			Text {
				text: qsTr("Page")
			}
			ComboBox {
				id: pageCombo
				model: pageListModel
				textRole: "name"
				currentIndex: -1
				onCurrentIndexChanged: changePage()
			}
			Button {
				text: qsTr("Reload")
				onClicked: reload()
			}
			CheckBox {
				id: autoReloadOnSave
				checked: true
				text: qsTr("Auto reload on save")
			}
		}

		WebEngineView {
			Layout.fillWidth: true
			Layout.fillHeight: true
			id: webView
			experimental.settings.localContentCanAccessFileUrls: true
			experimental.settings.localContentCanAccessRemoteUrls: true
			onJavaScriptConsoleMessage: {
				console.log(sourceID + ":" + lineNumber + ":" + message);
			}
			onLoadingChanged: {
				if (!loading) {
					initialized = true;
					webView.runJavaScript("init(\"" + socketServer.url + "\")");
					if (pendingPageUrl)
						setPreviewUrl(pendingPageUrl);
				}
			}
		}
	}
}
