import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtWebEngine 1.0
import QtWebEngine.experimental 1.0
import HttpServer 1.0
import "."

Item {
	id: webPreview
	property string pendingPageUrl: ""
	property bool initialized: false
	signal javaScriptMessage(var _level, string _sourceId, var _lineNb, string _content)

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
		if (initialized) {
			updateContract();
			//webView.runJavaScript("reloadPage()");
			setPreviewUrl(urlInput.text);
		}
	}

	function updateContract() {
		var contracts = {};
		for (var c in codeModel.contracts) {
			var contract = codeModel.contracts[c];
			contracts[c] = {
				name: contract.contract.name,
				address: clientModel.contractAddresses[contract.contract.name],
				interface: JSON.parse(contract.contractInterface),
			};
		}
		webView.runJavaScript("updateContracts(" + JSON.stringify(contracts) + ")");
	}

	function reloadOnSave() {
		if (autoReloadOnSave.checked)
			reload();
	}

	function updateDocument(documentId, action) {
		for (var i = 0; i < pageListModel.count; i++)
			if (pageListModel.get(i).documentId === documentId)
				action(i);
	}

	function changePage() {
		setPreviewUrl(urlInput.text);
		/*if (pageCombo.currentIndex >= 0 && pageCombo.currentIndex < pageListModel.count) {
			urlInput.text = httpServer.url + "/" + pageListModel.get(pageCombo.currentIndex).documentId;
			setPreviewUrl(httpServer.url + "/" + pageListModel.get(pageCombo.currentIndex).documentId);
		} else {
			setPreviewUrl("");
		}*/
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
					{
						urlInput.text = httpServer.url + "/" + document.documentId;
						setPreviewUrl(httpServer.url + "/" + document.documentId);
					}
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
				if (urlPath === "/")
					urlPath = "/index.html";
				var documentId = urlPath.substr(urlPath.lastIndexOf("/") + 1);
				var content = "";
				if (projectModel.codeEditor.isDocumentOpen(documentId))
					content = projectModel.codeEditor.getDocumentText(documentId);
				else
				{
					var doc = projectModel.getDocument(documentId);
					if (doc !== undefined)
						content = fileIo.readFile(doc.path);
				}

				if (documentId === urlInput.text.replace(httpServer.url + "/", "")) {
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
		spacing: 0
		Rectangle
		{
			anchors.leftMargin: 4
			color: WebPreviewStyle.general.headerBackgroundColor
			Layout.preferredWidth: parent.width
			Layout.preferredHeight: 32
			Row {
				anchors.top: parent.top
				anchors.fill: parent
				anchors.leftMargin: 3
				spacing: 3

				DefaultTextField
				{
					id: urlInput
					anchors.verticalCenter: parent.verticalCenter
					height: 21
					width: 300
					Keys.onEnterPressed:
					{
						setPreviewUrl(text);
					}
					Keys.onReturnPressed:
					{
						setPreviewUrl(text);
					}
					focus: true
				}

				Action {
					tooltip: qsTr("Reload")
					id: buttonReloadAction
					onTriggered: {
						reload();
					}
				}

				Button {
					iconSource: "qrc:/qml/img/available_updates.png"
					action: buttonReloadAction
					anchors.verticalCenter: parent.verticalCenter
					width: 21
					height: 21
					focus: true
				}

				Rectangle
				{
					width: 1
					height: parent.height - 10
					color: "#808080"
					anchors.verticalCenter: parent.verticalCenter
				}

				CheckBox {
					id: autoReloadOnSave
					checked: true
					height: 21
					anchors.verticalCenter: parent.verticalCenter
					style: CheckBoxStyle {
						label: DefaultLabel {
							text: qsTr("Auto reload on save")
						}
					}
					focus: true
				}

				Rectangle
				{
					width: 1
					height: parent.height - 10
					color: "#808080"
					anchors.verticalCenter: parent.verticalCenter
				}

				Button
				{
					height: 28
					anchors.verticalCenter: parent.verticalCenter
					action: expressionAction
					iconSource: "qrc:/qml/img/console.png"
				}

				Action {
					id: expressionAction
					tooltip: qsTr("Expressions")
					onTriggered:
					{
						expressionPanel.visible = !expressionPanel.visible;
						if (expressionPanel.visible)
						{
							webView.width = webView.parent.width - 350
							expressionInput.forceActiveFocus();
						}
						else
							webView.width = webView.parent.width
					}
				}
			}
		}

		SplitView
		{
			Layout.preferredWidth: parent.width
			Layout.fillHeight: true
			WebEngineView {
				Layout.fillHeight: true
				width: parent.width
				Layout.preferredWidth: parent.width
				id: webView
				experimental.settings.localContentCanAccessRemoteUrls: true
				onJavaScriptConsoleMessage: {
					webPreview.javaScriptMessage(level, sourceID, lineNumber, message);
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

			ListModel {
				id: javaScriptExpressionModel
			}

			Column {
				id: expressionPanel
				width: 350
				Layout.preferredWidth: 350
				Layout.fillHeight: true
				spacing: 0
				visible: false
				function addExpression()
				{
					if (expressionInput.text === "" || expressionInput.text === qsTr("Expression"))
						return;
					webView.runJavaScript("executeJavaScript(\"" + expressionInput.text.replace(/"/g, '\\"') + "\")", function(result) {
						javaScriptExpressionModel.insert(0, { expression: expressionInput.text, result: result });
						expressionInput.text = "";
					});
				}

				DefaultTextField {
					id: expressionInput
					width: parent.width
					height: 20
					font.family: "sans serif"
					font.italic: true
					font.pointSize: Style.absoluteSize(-3)
					Keys.onEnterPressed:
					{
						parent.addExpression();
					}

					Keys.onReturnPressed:
					{
						parent.addExpression();
					}

					onFocusChanged:
					{
						if (!focus && text == "")
							text = qsTr("Expression");
					}
					style: TextFieldStyle {
						background: Rectangle {
							color: "transparent"
						}
					}
				}

				TableView
				{
					width: parent.width
					height: webView.height - expressionInput.height
					model: javaScriptExpressionModel
					headerVisible: true
					rowDelegate:
						Rectangle {
						id: rowExpressions
						height: 20
						color: styleData.alternate ? "transparent" : "#f0f0f0"
					}

					onDoubleClicked:
					{
						var log = model.get(currentRow);
						if (log)
							appContext.toClipboard(log.expression + "\t" + log.result);
					}

					TableViewColumn {
						id: expression
						title: "Expression"
						role: "expression"
						width: 2 * (350 / 3)
						resizable: true
						delegate: Rectangle {
							color: "transparent"
							height: 20
							width: 2 * (350 / 3)
							MouseArea
							{
								anchors.fill: parent
								hoverEnabled: true
								onHoveredChanged:
								{
									deleteBtn.visible = containsMouse
								}
							}

							Button
							{
								id: deleteBtn
								iconSource: "qrc:/qml/img/delete_sign.png"
								action: deleteExpressionAction
								height: 18
								width: 18
								visible: false
							}

							Action {
								id: deleteExpressionAction
								tooltip: qsTr("Delete Expression")
								onTriggered:
								{
									javaScriptExpressionModel.remove(styleData.row);
								}
							}


							DefaultTextField {
								text:  styleData.value
								font.family: "sans serif"
								font.pointSize: Style.absoluteSize(-2)
								anchors.verticalCenter: parent.verticalCenter
								width: parent.width - deleteBtn.width
								anchors.left: deleteBtn.right
								anchors.leftMargin: 1

								MouseArea
								{
									anchors.fill: parent
									hoverEnabled: true
									onHoveredChanged:
									{
										deleteBtn.visible = containsMouse
									}
									onClicked:
									{
										parent.forceActiveFocus();
									}
								}

								function updateExpression()
								{
									if (text === "")
										javaScriptExpressionModel.remove(styleData.row);
									else
									{
										javaScriptExpressionModel.get(styleData.row).expression = text;
										webView.runJavaScript("executeJavaScript(\"" + text.replace(/"/g, '\\"') + "\")", function(result) {
											javaScriptExpressionModel.get(styleData.row).result = result;
										});
									}
								}

								Keys.onEnterPressed:
								{
									updateExpression();
								}

								Keys.onReturnPressed:
								{
									updateExpression();
								}

								style: TextFieldStyle {
									background: Rectangle {
										color: "transparent"
									}
								}
							}
						}
					}

					TableViewColumn {
						id: result
						title: "Result"
						role: "result"
						width: 350 / 3 - 5
						resizable: true
						delegate: Rectangle {
							color: "transparent"
							height: 20
							DefaultLabel {
								text: {
									var item = javaScriptExpressionModel.get(styleData.row);
									if (item !== undefined && item.result !== undefined)
										return item.result;
									else
										return "-";
								}
								font.family: "sans serif"
								font.pointSize: Style.absoluteSize(-2)
								anchors.verticalCenter: parent.verticalCenter
							}
						}
					}
				}
			}
		}
	}
}
