import QtQuick 2.2
import QtQuick.Controls 1.2
import QtQuick.Controls.Private 1.0 as ControlsPrivate
import QtQuick.Dialogs 1.2
import QtQuick.Dialogs.Private 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.1
import Qt.labs.folderlistmodel 2.1
import Qt.labs.settings 1.0

AbstractDialog {

    id: root

	property string folder: view.model.folder
	property var nameFilters: []
	property bool selectFolder: false
	property bool selectExisting: true
	property int selectedNameFilterIndex: -1
	property var selectedNameFilterExtensions: []
	property string selection: ""
	property alias fileUrl: root.selection


	function selectNameFilter(text) {
	}

	function clearSelection(text) {
		selection = "";
	}

	function addSelection(text) {
		selection = text;
	}

    onVisibleChanged: {
        if (visible) {
			view.needsWidthAdjustment = true
            view.selection.clear()
            view.focus = true
        }
	}

    Component.onCompleted: {
		folder = fileIo.pathToUrl(fileIo.homePath);
		view.model.nameFilters = root.selectedNameFilterExtensions
        filterField.currentIndex = root.selectedNameFilterIndex
        root.favoriteFolders = settings.favoriteFolders
    }

    Component.onDestruction: {
        settings.favoriteFolders = root.favoriteFolders
    }

    property Settings settings: Settings {
        category: "QQControlsFileDialog"
        property alias width: root.width
        property alias height: root.height
		property variant favoriteFolders: []
    }

    property bool showFocusHighlight: false
    property SystemPalette palette: SystemPalette { }
    property var favoriteFolders: []

    function dirDown(path) {
        view.selection.clear()
        root.folder = "file://" + path
    }
    function dirUp() {
		view.selection.clear()
        if (view.model.parentFolder != "")
			root.folder = view.model.parentFolder
	}
    function acceptSelection() {
        // transfer the view's selections to QQuickFileDialog
        clearSelection()
        if (selectFolder && view.selection.count === 0)
            addSelection(folder)
        else {
            view.selection.forEach(function(idx) {
                if (view.model.isFolder(idx)) {
                    if (selectFolder)
                        addSelection(view.model.get(idx, "fileURL"))
                } else {
                    if (!selectFolder)
                        addSelection(view.model.get(idx, "fileURL"))
                }
            })
        }
        accept()
    }

    property Action dirUpAction: Action {
        text: "\ue810"
        shortcut: "Ctrl+U"
        onTriggered: dirUp()
        tooltip: qsTr("Go up to the folder containing this one")
    }

    Rectangle {
        id: window
		implicitWidth: Math.min(root.__maximumDimension, Math.max(Screen.pixelDensity * 100, view.implicitWidth))
        implicitHeight: Math.min(root.__maximumDimension, Screen.pixelDensity * 80)
        color: root.palette.window

        Binding {
            target: view.model
            property: "folder"
            value: root.folder
        }
        Binding {
            target: currentPathField
            property: "text"
			value: fileIo.urlToPath(root.folder)
        }
        Keys.onPressed: {
            event.accepted = true
            switch (event.key) {
            case Qt.Key_Back:
            case Qt.Key_Escape:
                reject()
                break
            default:
                event.accepted = false
                break
            }
        }
        Keys.forwardTo: [view.flickableItem]


		TableView {
			id: view
			sortIndicatorVisible: true
			width: parent.width
			anchors.top: titleBar.bottom
			anchors.bottom: bottomBar.top

			property bool needsWidthAdjustment: true
			selectionMode: root.selectMultiple ?
				(ControlsPrivate.Settings.hasTouchScreen ? SelectionMode.MultiSelection : SelectionMode.ExtendedSelection) :
				SelectionMode.SingleSelection
			onRowCountChanged: if (needsWidthAdjustment && rowCount > 0) {
				resizeColumnsToContents()
				needsWidthAdjustment = false
			}
			model: FolderListModel {
				showFiles: !root.selectFolder
				nameFilters: root.selectedNameFilterExtensions
				sortField: (view.sortIndicatorColumn === 0 ? FolderListModel.Name :
							(view.sortIndicatorColumn === 1 ? FolderListModel.Type :
							(view.sortIndicatorColumn === 2 ? FolderListModel.Size : FolderListModel.LastModified)))
				sortReversed: view.sortIndicatorOrder === Qt.DescendingOrder
			}

			onActivated: if (view.focus) {
				if (view.selection.count > 0 && view.model.isFolder(row)) {
					dirDown(view.model.get(row, "filePath"))
				} else {
					root.acceptSelection()
				}
			}
			onClicked: currentPathField.text = view.model.get(row, "filePath")


			TableViewColumn {
				id: fileNameColumn
				role: "fileName"
				title: qsTr("Filename")
				delegate: Item {
					implicitWidth: pathText.implicitWidth + pathText.anchors.leftMargin + pathText.anchors.rightMargin

					Text {
						id: fileIcon
						width: height
						verticalAlignment: Text.AlignVCenter
						font.family: iconFont.name
						property alias unicode: fileIcon.text
						FontLoader { id: iconFont; source: "qrc:/QtQuick/Dialogs/qml/icons.ttf"; onNameChanged: console.log("custom font" + name) }
						x: 4
						height: parent.height - 2
						unicode: view.model.isFolder(styleData.row) ? "\ue804" : "\ue802"
					}
					Text {
						id: pathText
						text: styleData.value
						anchors {
							left: parent.left
							right: parent.right
							leftMargin: 36 + 6
							rightMargin: 4
							verticalCenter: parent.verticalCenter
						}
						color: styleData.textColor
						elide: Text.ElideRight
						renderType: ControlsPrivate.Settings.isMobile ? Text.QtRendering : Text.NativeRendering
					}
				}
			}
			TableViewColumn {
				role: "fileSuffix"
				title: qsTr("Type", "file type (extension)")
				// TODO should not need to create a whole new component just to customize the text value
				// something like textFormat: function(text) { return view.model.get(styleData.row, "fileIsDir") ? "folder" : text }
				delegate: Item {
					implicitWidth: sizeText.implicitWidth + sizeText.anchors.leftMargin + sizeText.anchors.rightMargin
					Text {
						id: sizeText
						text: view.model.get(styleData.row, "fileIsDir") ? "folder" : styleData.value
						anchors {
							left: parent.left
							right: parent.right
							leftMargin: 4
							rightMargin: 4
							verticalCenter: parent.verticalCenter
						}
						color: styleData.textColor
						elide: Text.ElideRight
						renderType: ControlsPrivate.Settings.isMobile ? Text.QtRendering : Text.NativeRendering
					}
				}
			}
			TableViewColumn {
				role: "fileSize"
				title: qsTr("Size", "file size")
				horizontalAlignment: Text.AlignRight
			}
			TableViewColumn { id: modifiedColumn; role: "fileModified" ; title: qsTr("Modified", "last-modified time") }
			TableViewColumn { id: accessedColumn; role: "fileAccessed" ; title: qsTr("Accessed", "last-accessed time") }
		}

        ToolBar {
            id: titleBar
            RowLayout {
                anchors.fill: parent
                ToolButton {
                    action: dirUpAction
					//style: IconButtonStyle { }
                    Layout.maximumWidth: height * 1.5
                }
                TextField {
                    id: currentPathField
                    Layout.fillWidth: true
                    function doAccept() {
                        root.clearSelection()
						if (root.addSelection(fileIo.pathToUrl(text)))
                            root.accept()
                        else
							root.folder = fileIo.pathFolder(text)
                    }
                    onAccepted: doAccept()
                }
            }
        }
        Item {
            id: bottomBar
            width: parent.width
            height: buttonRow.height + buttonRow.spacing * 2
            anchors.bottom: parent.bottom

            Row {
                id: buttonRow
                anchors.right: parent.right
                anchors.rightMargin: spacing
                anchors.verticalCenter: parent.verticalCenter
                spacing: 4
				ComboBox {
                    id: filterField
                    model: root.nameFilters
                    visible: !selectFolder
					width: bottomBar.width - cancelButton.width - okButton.width - parent.spacing * 6
                    anchors.verticalCenter: parent.verticalCenter
                    onCurrentTextChanged: {
                        root.selectNameFilter(currentText)
                        view.model.nameFilters = root.selectedNameFilterExtensions
                    }
                }
                Button {
                    id: cancelButton
                    text: qsTr("Cancel")
                    onClicked: root.reject()
                }
                Button {
                    id: okButton
                    text: root.selectFolder ? qsTr("Choose") : (selectExisting ? qsTr("Open") : qsTr("Save"))
                    onClicked: {
                        if (view.model.isFolder(view.currentIndex) && !selectFolder)
                            dirDown(view.model.get(view.currentIndex, "filePath"))
                        else if (!(root.selectExisting))
                            currentPathField.doAccept()
                        else
                            root.acceptSelection()
                    }
                }
            }
        }
    }
}
