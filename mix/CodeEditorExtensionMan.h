#ifndef CODEEDITOREXTENSIONMANAGER_H
#define CODEEDITOREXTENSIONMANAGER_H

#include <QQuickItem>
#include <QTextDocument>
#include "Feature.h"

class CodeEditorExtensionManager : public QObject
{
    Q_OBJECT

    Q_PROPERTY(QQuickItem* editor MEMBER m_editor WRITE setEditor)
    Q_PROPERTY(QQuickItem* tabView MEMBER m_tabView WRITE setTabView)

public:
    CodeEditorExtensionManager();
    void initExtensions();
    void setEditor(QQuickItem*);
    void setTabView(QQuickItem*);

private:
    QQuickItem* m_editor;
    QQuickItem* m_tabView;
    QTextDocument* m_doc;
    void loadEditor(QQuickItem*);
};

#endif // CODEEDITOREXTENSIONMANAGER_H
