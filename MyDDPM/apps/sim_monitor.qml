import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtCharts 2.15
import Qt.labs.folderlistmodel 1.0
//import "sim_utils.js" as Utils

ApplicationWindow {
    visible: true
    width: 600
    height: 600
    title: "Sampling Sequence"
    SwipeView {
        id: view
        height: parent.height-64
        width: parent.width
        currentIndex: 0
        // View 0
        Item {
            id: view_0
            ImageGrid {
                height: 200
                width: 200
                property var page: pages_model["page0"]
                n_col:page["grid_n_col"]
                n_row:page["grid_n_row"]
                imagePaths: page["image_path_list"]
                info: page["info"]
            }

        }
        // View 1
        Item {
            id: view_1
            ImageGrid {
                height: 200
                width: 200
                property var page: pages_model["page1"]
                n_col:page["grid_n_col"]
                n_row:page["grid_n_row"]
                imagePaths: page["image_path_list"]
                info: page["info"]
            }

        }
    }
    PageIndicator {
        id: indicator

        count: view.count
        currentIndex: view.currentIndex

        anchors.bottom: view.bottom
        anchors.horizontalCenter: parent.horizontalCenter
    }
}