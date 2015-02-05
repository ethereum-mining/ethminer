/*
 * Used to instanciate a QEther obj using Qt.createComponent function.
*/
import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Controls.Styles 1.1
import org.ethereum.qml.QEther 1.0

QEther
{
	id: basicEther
	value: "100000000000"
	unit: QEther.Wei
}
