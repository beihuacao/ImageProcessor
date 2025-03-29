#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QAction>
#include <QDebug>
#include <QMessageBox>
#include <QFileDialog>
#include <QImage>
#include <QButtonGroup>
#include <QIntValidator>
#include <QSlider>

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <fstream>

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>


using namespace std;
using namespace cv;



QT_BEGIN_NAMESPACE
namespace Ui
{
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

    bool process_mode = false;
    Mat image_raw, image_result, image_reduced;
    QImage image_qt, image_qt_result, image_qt_reduced;
    Mat gradXY, theta;
    QString filename;
    float reduce_scale;
    bool large = false;
    Mat RGB2GRAY(Mat img);
    Mat RGB2HSV(Mat img);
    Mat Horizontal_Mirroring(Mat image);
    Mat Vertical_Mirroring(Mat image);
    Mat Rotate(Mat image, float deg);
    Mat Threshold_Segmentation(Mat image, int threshold);
    int dajin(Mat image);
    Mat Reverse(Mat image);
    Mat Erosion(Mat image);
    Mat Dilation(Mat image);
    Mat Histogram_Equalization(Mat image);
    Mat Scaling(Mat image, float scale);

    Mat Mean_Filtering(Mat image, int kernal);
    Mat Median_Filtering(Mat image, int kernal);
    Mat Gaussian_Filtering(Mat image, int kernal);
    Mat Sobel_Filtering(Mat image, int kernal);
    Mat Laplace_Filtering(Mat image);


    void Canny_Filtering(Mat img);
    Mat doubleThreshold(Mat image);
    Mat nonLocalMaxValue(Mat gradXY, Mat theta);
    void getGrandient(Mat img);



private slots:

    void openImage();

    void on_start_clicked();
    void on_image_raw_clicked();
    void on_image_result_clicked();
    void on_image_trans_clicked();
    void on_image_filter_clicked();
    void on_save_result_clicked();

    bool check(QImage image_qt);
    void on_gray_clicked();
    void on_hsv_clicked();

    void on_horizon_clicked();
    void on_vertical_clicked();

    void on_rotate_slider_valueChanged(int value);

    void on_default_segmentation_clicked();
    void on_color_segmentation_clicked();
    void on_dajin_clicked();

    void on_reverse_2_clicked();
    void on_reverse_gray_clicked();

    void on_Erosion_2_clicked();
    void on_Dilation_2_clicked();
    void on_open_clicked();
    void on_close_clicked();

    void on_histograam_clicked();
    void on_hist_color_clicked();

    void on_horizontalSlider_valueChanged(int value);

    void on_mean_3_clicked();
    void on_mean_5_clicked();
    void on_mean_7_clicked();

    void on_medium_3_clicked();
    void on_medium_5_clicked();
    void on_medium_7_clicked();

    void on_Gaussian_3_clicked();
    void on_Gaussian_5_clicked();

    void on_sobel_x_clicked();
    void on_sobel_y_clicked();
    void on_sobel_abs_clicked();

    void on_laplace_1_clicked();
    void on_laplace_reduce_clicked();
    void on_laplace_edge_clicked();

    void on_canny_1_clicked();

    void on_noise_clicked();


    void on_opencv_mode_clicked();
    void on_raw_mode_clicked();
private:
    Ui::MainWindow* ui;
};
#endif   // MAINWINDOW_H
