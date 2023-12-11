import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import Qt.labs.folderlistmodel 1.0

Item {
    id: mainTop
    property int n_row: 4
    property int n_col: 4
    property int cellHeight: 64
    property int cellWidth: 64
    property var imagePaths: []
    property string info: "No info."
    property int infoHeight: 32

    Rectangle{
        anchors.fill: parent
        border.width: 1
    Column {
        anchors.fill: parent
        Rectangle{
            color: "lightgray"
            height: mainTop.infoHeight
            width: parent.width
            ScrollView{
                anchors.fill: parent
                TextArea {
                    id: infoText
                    height: mainTop.infoHeight
                    width: parent.width
                    text: mainTop.info
                }
            }
        }
        GridView {
            id: gridView
            height: parent.height-infoText.height
            width: parent.width
            model: mainTop.n_row*mainTop.n_col
            delegate: Item {
                height: mainTop.cellHeight
                width: mainTop.cellWidth
                Loader {
                    sourceComponent: Image {
                        height: mainTop.cellHeight
                        width: mainTop.cellWidth
                        source: mainTop.imagePaths[index]
                        fillMode: Image.PreserveAspectFit
                    }
                }
            }
        }
    }
    }
}