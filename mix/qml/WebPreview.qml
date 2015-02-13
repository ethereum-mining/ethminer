import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0
import QtQuick.Controls.Styles 1.1
import QtWebEngine 1.0
import QtWebEngine.experimental 1.0
import HttpServer 1.0

Item {
	id: webPreview
	property string pendingPageUrl: ""
	property bool initialized: false

	function setPreviewUrl(url) {
		if (!initialized)
			pendingPageUrl = url;
		else {
			pendingPageUrl = "";
			updateContract();
			webView.runJavaScript("loadPage(\"" + url + "\")");
		}
	}

	function reload() {
		updateContract();
		webView.runJavaScript("reloadPage()");
	}

	function updateContract() {
		webView.runJavaScript("updateContract(\"" + codeModel.code.contract.name + "\", \"" + clientModel.contractAddress + "\", " + codeModel.code.contractInterface + ")");
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
		if (pageCombo.currentIndex >= 0 && pageCombo.currentIndex < pageListModel.count) {
			setPreviewUrl(httpServer.url + "/" + pageListModel.get(pageCombo.currentIndex).documentId);
		} else {
			setPreviewUrl("");
		}
	}
	Connections {
		target: appContext
		onAppLoaded: {
			//We need to load the container using file scheme so that web security would allow loading local files in iframe
			var containerPage = fileIo.readFile("qrc:///qml/html/WebContainer.html");
			webView.loadHtml(containerPage, httpServer.url + "/WebContainer.html")

		}
	}

	Connections {
		target: clientModel
		onContractAddressChanged: reload();
		onRunComplete: reload();
	}

	Connections {
		target: codeModel
		onContractInterfaceChanged: reload();
	}

	Connections {
		target: projectModel
		//onProjectSaved : reloadOnSave();
		//onDocumentSaved: reloadOnSave();
		onDocumentAdded: {
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

		onProjectLoading: {
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

	HttpServer {
		id: httpServer
		listen: true
		accept: true
		port: 8893
		onClientConnected: {
			var urlPath = _request.url.toString();
			if (urlPath.indexOf("/rpc/") === 0)
			{
				//jsonrpc request
				//filter polling requests //TODO: do it properly
				var log = _request.content.indexOf("eth_changed") < 0;
				if (log)
					console.log(_request.content);
				var response = clientModel.apiCall(_request.content);
				if (log)
					console.log(response);
				_request.setResponse(response);
			}
			else
			{
				//document request
				var documentId = urlPath.substr(urlPath.lastIndexOf("/") + 1);
				var content = "";
				if (projectModel.codeEditor.isDocumentOpen(documentId))
					content = projectModel.codeEditor.getDocumentText(documentId);
				else
					content = fileIo.readFile(projectModel.getDocument(documentId).path);
				if (documentId === pageListModel.get(pageCombo.currentIndex).documentId) {
					//root page, inject deployment script
					content = "<script>web3=parent.web3;contracts=parent.contracts;</script>\n" + content;
					_request.setResponseContentType("text/html");
				}
				_request.setResponse(content);
			}
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
			experimental.settings.localContentCanAccessRemoteUrls: true
			onJavaScriptConsoleMessage: {
				console.log(sourceID + ":" + lineNumber + ":" + message);
			}
			onLoadingChanged: {
				if (!loading) {
					initialized = true;
					webView.runJavaScript("init(\"" + httpServer.url + "/rpc/\")");
					if (pendingPageUrl)
						setPreviewUrl(pendingPageUrl);
				}
			}
		}
	}
}
