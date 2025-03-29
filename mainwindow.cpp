#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "opencvImageProcessor.h"

cv::Mat QImage2cvMat(QImage &image, bool rb_swap)
{
    cv::Mat mat;
    switch(image.format())
    {
        case QImage::Format_RGB888:
            mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void *)image.constBits(), image.bytesPerLine());
            mat = mat.clone();
            if(rb_swap) cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
            break;
        case QImage::Format_Indexed8:
        case QImage::Format_Grayscale8:
            mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void *)image.bits(), image.bytesPerLine());
            mat = mat.clone();
            break;
        case QImage::Format_ARGB32:
        case QImage::Format_RGB32:
        case QImage::Format_ARGB32_Premultiplied:
            auto mat_tmp = cv::Mat(image.height(), image.width(), CV_8UC4, (void *)image.constBits(), image.bytesPerLine());
            auto img=cv::Mat(image.height(), image.width(), CV_8UC3, cv::Scalar(0));
            cv::cvtColor(mat_tmp , img , cv::COLOR_RGBA2RGB);
            mat = img.clone();
            break;
    }
    return mat;
}

QImage cvMat2QImage(const cv::Mat& mat, bool rb_swap)
{
    const uchar *pSrc = (const uchar*)mat.data;
    if(mat.type() == CV_8UC1)
    {
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
        return image.copy();

    }
    else if(mat.type() == CV_8UC3)
    {
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        if(rb_swap) return image.rgbSwapped();
        return image.copy();
    }
    else
    {
        qDebug() << "ERROR: Mat could not be converted to QImage.";
        return QImage();
    }
}


Mat colcon(Mat image, vector<vector<float>> kernal) {
    Mat img;
    if(image.channels() == 1)
        img = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
    else
        img = cv::Mat(image.rows, image.cols, CV_8UC3, cv::Scalar(0));

    int width = img.cols;
    int height = img.rows;

    for(int row=0; row<height; ++row) {
        for(int col=0; col<width; ++col) {
            if(kernal.size() == 3) {
                float tmp = 0.0;
                for(int i=-1; i<=1; ++i) {
                    for(int j=-1; j<=1; ++j) {
                        if(row+i>=0 && row+i<height && col+j>=0 && col+j<width)
                            tmp += kernal[i+1][j+1] * (float)image.at<uchar>(row+i, col+j);
                    }
                }
                img.at<uchar>(row,col) = (tmp>0)?(int)tmp:0;
            }
            else if(kernal.size() == 5) {
                float tmp = 0.0;
                for(int i=-2; i<=2; ++i) {
                    for(int j=-2; j<=2; ++j) {
                        if(row+i>=0 && row+i<height && col+j>=0 && col+j<width)
                            tmp += kernal[i+2][j+2] * (float)image.at<uchar>(row+i, col+j);
                    }
                }
                img.at<uchar>(row,col) = (tmp>0)?(int)tmp:0;
            }
            else {
                float tmp = 0.0;
                for(int i=-3; i<=3; ++i) {
                    for(int j=-3; j<=3; ++j) {
                        if(row+i>=0 && row+i<height && col+j>=0 && col+j<width)
                            tmp += kernal[i+3][j+3] * (float)image.at<uchar>(row+i, col+j);
                    }
                }
                img.at<uchar>(row,col) = (tmp>0)?(int)tmp:0;
            }
        }
    }
    return img;
}

Mat addSaltNoise(const Mat srcImage, int n) {
    Mat dstImage = srcImage.clone();
    for(int k=0; k<n; k++) {
        int i = rand() % dstImage.rows;
        int j =rand() % dstImage.cols;
        if(dstImage.channels() == 1)
            dstImage.at<uchar>(i,j) = 255;
        else {
            dstImage.at<Vec3b>(i,j)[0] = 255;
            dstImage.at<Vec3b>(i,j)[1] = 255;
            dstImage.at<Vec3b>(i,j)[2] = 255;
        }
    }

    for(int k=0; k<n; ++k) {
        int i = rand() % dstImage.rows;
        int j = rand() % dstImage.cols;
        if(dstImage.channels() == 1) {
            dstImage.at<uchar>(i,j) = 0;
        }
        else {
            dstImage.at<Vec3b>(i,j)[0] = 0;
            dstImage.at<Vec3b>(i,j)[1] = 0;
            dstImage.at<Vec3b>(i,j)[2] = 0;
        }
    }
    return dstImage;
}


MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->action_open, &QAction::triggered, this, &MainWindow::openImage);

    QButtonGroup* buttonGroup = new QButtonGroup(this);
    buttonGroup->addButton(ui->gray);
    buttonGroup->addButton(ui->hsv);
    buttonGroup->addButton(ui->horizon);
    buttonGroup->addButton(ui->vertical);
    buttonGroup->addButton(ui->default_segmentation);
    buttonGroup->addButton(ui->color_segmentation);
    buttonGroup->addButton(ui->dajin);
    buttonGroup->addButton(ui->reverse_2);
    buttonGroup->addButton(ui->reverse_gray);
    buttonGroup->addButton(ui->Erosion_2);
    buttonGroup->addButton(ui->Dilation_2);
    buttonGroup->addButton(ui->open);
    buttonGroup->addButton(ui->close);
    buttonGroup->addButton(ui->histograam);
    buttonGroup->addButton(ui->hist_color);


    QButtonGroup* buttonGroup2 = new QButtonGroup(this);
    buttonGroup2->addButton(ui->mean_3);
    buttonGroup2->addButton(ui->mean_5);
    buttonGroup2->addButton(ui->mean_7);
    buttonGroup2->addButton(ui->medium_3);
    buttonGroup2->addButton(ui->medium_5);
    buttonGroup2->addButton(ui->medium_7);
    buttonGroup2->addButton(ui->Gaussian_3);
    buttonGroup2->addButton(ui->Gaussian_5);
    buttonGroup2->addButton(ui->sobel_abs);
    buttonGroup2->addButton(ui->sobel_x);
    buttonGroup2->addButton(ui->sobel_y);
    buttonGroup2->addButton(ui->laplace_1);
    buttonGroup2->addButton(ui->laplace_reduce);
    buttonGroup2->addButton(ui->laplace_edge);

    QButtonGroup* buttonGroup3 = new QButtonGroup(this);
    buttonGroup3->addButton(ui->opencv_mode);
    buttonGroup3->addButton(ui->raw_mode);

    filename = "/home/cdy/qtProject/ImageProcessor_Learn/images/lena1.jpg";
    if( !(image_qt.load(filename) )) {
        qDebug() << "打开图像失败";
        return;
    }
    if(check(image_qt) == true) {
        qDebug() << "加载了一张图片";
    }


    ui->rotate_slider->setMinimum(-90);
    ui->rotate_slider->setMaximum(90);
    ui->rotate_slider->setSingleStep(1);
    ui->rotate_slider->setTickInterval(1);
    ui->rotate_slider->setValue(0);

    ui->horizontalSlider->setMinimum(10);
    ui->horizontalSlider->setMaximum(150);
    ui->horizontalSlider->setSingleStep(1);
    ui->horizontalSlider->setTickInterval(1);
    ui->horizontalSlider->setValue(100);

    ui->rotate_slider->setTickPosition(QSlider::TicksBelow);
    ui->horizontalSlider->setTickPosition(QSlider::TicksBelow);
    ui->label->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    ui->label_2->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    ui->label_3->setAlignment(Qt::AlignTop | Qt::AlignLeft);
}

MainWindow::~MainWindow()
{
    delete ui;
}





bool MainWindow::check(QImage image_qt) {
    if(image_qt.width() > 750 || image_qt.height() > 520) {
        large = true;
        if( (float)image_qt.width() / (float)image_qt.height() >= 750.0/520.0 ) {
            reduce_scale = 750 / (float)image_qt.width();
        } else {
            reduce_scale = 520 / (float)image_qt.height();
        }
        image_qt_reduced = image_qt.scaled( (int)(image_qt.width() * reduce_scale - 1),
                                           (int)(image_qt.height() * reduce_scale - 1),
                                           Qt::KeepAspectRatio, Qt::SmoothTransformation);
        image_raw = QImage2cvMat(image_qt_reduced, false);
        ui->label_2->setPixmap(QPixmap::fromImage(image_qt_reduced));
        ui->stackedWidget_2->setCurrentIndex(0);
        return true;
    }
    image_raw = QImage2cvMat(image_qt, false);
    ui->label_2->setPixmap(QPixmap::fromImage(image_qt));
    ui->stackedWidget_2->setCurrentIndex(0);
    return false;
}


void MainWindow::openImage() {
    filename = QFileDialog::getOpenFileName(this, "选择图片", QCoreApplication::applicationDirPath(),
                                            "Images (*.png *.bmp *.jpg *.jpeg)");
    if(filename.isEmpty())
        return;
    if(!(image_qt.load(filename))) {
        qDebug() << "打开图像失败";
        return;
    }
    if(check(image_qt) == false) {
        ui->label_2->setPixmap(QPixmap::fromImage(image_qt));
        ui->stackedWidget_2->setCurrentIndex(0);
    }
}

// 转为灰度图
Mat MainWindow::RGB2GRAY(Mat img) {
    int width = img.cols;
    int height = img.rows;
    cv::Mat grayImage(height, width, CV_8UC1, cv::Scalar(0));
    if(img.channels() == 1)
        return img.clone();
    else {
        for(int row = 0; row < height; row++) {
            for(int col=0; col < width; col++) {
                Vec3b bgr = img.at<Vec3b>(row, col);
                grayImage.at<uchar>(row,col)=0.1140*bgr[0]+0.5870*bgr[1]+0.2989*bgr[2];
            }
        }
    }
    return grayImage;
}

// 转为hsv
Mat MainWindow::RGB2HSV(Mat img) {
    auto hsvImage = cv::Mat(img.size(), CV_8UC3, cv::Scalar(0,0,0));
    for(int i=0;  i < img.rows; ++i) {
        for(int j=0; j< img.cols; ++j) {
            auto& rgb = img.at<Vec3b>(i,j);
            float r = rgb[2] / 255.0, g = rgb[1] / 255.0, b = rgb[0] / 255.0;
            float h, s, v;
            float minVal = std::min({r,g,b});
            float maxVal = std::max({r,g,b});

            v = maxVal;
            s = (maxVal==0)?0:(1-minVal/maxVal);
            float delta = maxVal - minVal;
            if(delta == 0)
                h = 0;
            else {
                if(maxVal == r && g >= b)
                    h = 60 * (g-b)/(maxVal - minVal);
                else if(maxVal == r && g < b)
                    h = 60 * (g-b)/(maxVal - minVal) + 360;
                else if(maxVal == g)
                    h = 120 + 60*(b-r)/(maxVal - minVal);
                else
                    h = 240 + 60*(r-g)/(maxVal - minVal);
            }
            hsvImage.at<Vec3b>(i,j)[0] = (int)(h*255/360);
            hsvImage.at<Vec3b>(i,j)[1] = (int)(s*255);
            hsvImage.at<Vec3b>(i,j)[2] = (int)(v*255);
        }
    }
    return hsvImage;
}

// 水平镜像
Mat MainWindow::Horizontal_Mirroring(Mat image) {
    auto img = image.clone();
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();

    for(int row = 0; row < height; ++row) {
        for(int col = 0; col < width; ++col) {
            if(channels == 1)
                img.at<uchar>(row, width-col-1) = image.at<uchar>(row,col);
            if(channels == 3)
                img.at<Vec3b>(row, width-col-1) = image.at<Vec3b>(row,col);
        }
    }
    return img;
}

// 垂直镜像
Mat MainWindow::Vertical_Mirroring(Mat image) {
    auto img = image.clone();
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();

    for(int row = 0; row < height; ++row) {
        for(int col = 0; col < width; ++col) {
            if(channels == 1)
                img.at<uchar>(height-row-1, col) = image.at<uchar>(row,col);
            if(channels == 3)
                img.at<Vec3b>(height-row-1, col) = image.at<Vec3b>(row,col);
        }
    }
    return img;
}

vector<float> rotate(vector<int> point, vector<int> center, float degree) {
    vector<float> result(2), temp(2);
    temp[0] = (float)(point[0] - center[0]);
    temp[1] = (float)(point[1] - center[1]);
    auto rad = degree * M_PI / 180.0;
    result[0] = cos(rad)*temp[0] - sin(rad)*temp[1] + (float)center[0];
    result[1] = sin(rad)*temp[0] + cos(rad)*temp[1] + (float)center[1];
    return result;
}


vector<float> rotate_inv(vector<int> point, vector<int> center, float degree) {
    vector<float> result(2), temp(2);
    temp[0] = (float)(point[0] - center[0]);
    temp[1] = (float)(point[1] - center[1]);
    auto rad = degree * M_PI / 180.0;
    result[0] = cos(rad)*temp[0] + sin(rad)*temp[1] + (float)center[0];
    result[1] = -sin(rad)*temp[0] + cos(rad)*temp[1] + (float)center[1];
    return result;
}


Mat MainWindow::Rotate(Mat image, float deg) {
    auto width = image.cols;
    auto height = image.rows;
    vector<int> left_up(2), left_down(2), right_up(2), right_down(2);
    left_up={0,0}, left_down={height,0}, right_up={0,width}, right_down={height,width};
    auto rotated_left_up = rotate(left_up, left_down, deg);
    auto rotated_left_down = rotate(left_down, left_down, deg);
    auto rotated_right_up = rotate(right_up, left_down, deg);
    auto rotated_right_down = rotate(right_down, left_down, deg);
    int h = (int)std::max({rotated_left_up[0],rotated_left_down[0],rotated_right_up[0], rotated_right_down[0]}) -
                (int)std::min({rotated_left_up[0],rotated_left_down[0],rotated_right_up[0], rotated_right_down[0]});
    int w = (int)std::max({rotated_left_up[1],rotated_left_down[1],rotated_right_up[1], rotated_right_down[1]}) -
                (int)std::min({rotated_left_up[1],rotated_left_down[1],rotated_right_up[1], rotated_right_down[1]});

    cv::Mat img;
    int min_h = (int)std::min({rotated_left_up[0],rotated_left_down[0],rotated_right_up[0],rotated_right_down[0]});
    int min_w = (int)std::min({rotated_left_up[1],rotated_left_down[1],rotated_right_up[1],rotated_right_down[1]});

    if(image.channels() == 1) {
        img = cv::Mat(h, w, CV_8UC1, cv::Scalar(0));
        for(int i=0; i<h; ++i) {
            for(int j=0; j<w; ++j) {
                vector<int> before_trans(2);
                before_trans[0] = i + min_h;
                before_trans[1] = j + min_w;
                auto before_rotate = rotate_inv(before_trans, left_down, deg);
                if(before_rotate[0]<0 || before_rotate[0]>=height || before_rotate[1]<0 || before_rotate[1]>=width)
                    img.at<uchar>(i,j) = 0;
                else {
                    int floor_height = std::floor(before_rotate[0]), floor_width = std::floor(before_rotate[1]),
                        ceil_height = std::ceil(before_rotate[0]), ceil_width = std::ceil(before_rotate[1]);
                    float upper_rate = (float)ceil_height-before_rotate[0],
                            low_rate = 1 - upper_rate,
                            left_rate= (float)ceil_width - before_rotate[1],
                            right_rate = 1 - left_rate;
                    img.at<uchar>(i,j) = (int)(upper_rate*(left_rate*image.at<uchar>(floor_height,floor_width)+right_rate*image.at<uchar>(floor_height,ceil_width))
                                                + low_rate * (left_rate*image.at<uchar>(ceil_height,floor_width)+right_rate*image.at<uchar>(ceil_height,ceil_width)));

                }
            }
        }
    }
    else if(image.channels() == 3) {
        img = cv::Mat(h, w, CV_8UC3, cv::Scalar(0));
        for(int i=0; i<h; ++i) {
            for(int j=0; j<w; ++j) {
                vector<int> before_trans(2);
                before_trans[0] = i + min_h;
                before_trans[1] = j + min_w;
                auto before_rotate = rotate_inv(before_trans, left_down, deg);

                if(before_rotate[0]>=0 && before_rotate[0]<height && before_rotate[1]>=0 && before_rotate[1]<width) {
                    int floor_height = std::floor(before_rotate[0]), floor_width = std::floor(before_rotate[1]),
                        ceil_height = std::ceil(before_rotate[0]), ceil_width = std::ceil(before_rotate[1]);
                    float upper_rate = (float)ceil_height-before_rotate[0],
                        low_rate = 1 - upper_rate,
                        left_rate= (float)ceil_width - before_rotate[1],
                        right_rate = 1 - left_rate;
                    img.at<Vec3b>(i,j) = (Vec3b)(upper_rate*(left_rate*image.at<Vec3b>(floor_height,floor_width)+right_rate*image.at<Vec3b>(floor_height,ceil_width))
                                                + low_rate * (left_rate*image.at<Vec3b>(ceil_height,floor_width)+right_rate*image.at<Vec3b>(ceil_height,ceil_width)));
                }
            }
        }
    }
    return img;
}

Mat MainWindow::Threshold_Segmentation(Mat image, int threshold) {
    auto img = image.clone();
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    for(int row = 0; row < height; ++row) {
        for(int col = 0; col < width; ++col) {
            if(channels == 1) {
                int pv = img.at<uchar>(row,col);
                if(pv >= threshold)
                    img.at<uchar>(row,col) = 255;
                else
                    img.at<uchar>(row,col) = 0;
            }
            if(channels == 3) {
                Vec3b bgr = img.at<Vec3b>(row,col);
                if(bgr[0] >= threshold)
                    img.at<Vec3b>(row, col)[0] = 255;
                else
                    img.at<Vec3b>(row,col)[0] = 0;
                if(bgr[1] >= threshold)
                    img.at<Vec3b>(row, col)[1] = 255;
                else
                    img.at<Vec3b>(row,col)[1] = 0;
                if(bgr[2] >= threshold)
                    img.at<Vec3b>(row, col)[2] = 255;
                else
                    img.at<Vec3b>(row,col)[2] = 0;
            }
        }
    }
    return img;
}

int MainWindow::dajin(Mat image) {
    int width = image.cols;
    int height = image.rows;

    vector<int> mean_temp(256,0);
    vector<int> p(256,0);
    vector<float> pp(256,0);
    vector<float> variance_temp(256,0.0);
    int mean_all = 0;
    for(int row = 0; row < height; ++row) {
        for(int col = 0; col < width; ++col) {
            for(int i = 255; i >= image.at<uchar>(row,col); --i) {
                mean_temp[i] += image.at<uchar>(row,col);
                p[i] += 1;
            }
        }
    }
    mean_all = mean_temp[255] / p[255];
    for(int i=1; i<255; ++i) {
        if(p[i] == 0) {
            variance_temp[i] = -9999;
            continue;
        }
        variance_temp[i] = p[i] * (mean_all - mean_temp[i] / p[i]) * (mean_all - mean_temp[i] / p[i]) / (width * height - p[i]);
    }
    auto max_iter = std::max_element(variance_temp.begin(), variance_temp.end());
    int max_index = (int)std::distance(variance_temp.begin(), max_iter);

    return max_index;

}


Mat MainWindow::Reverse(Mat image) {
    auto img = image.clone();
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    for(int row=0; row<height; ++row) {
        for(int col=0; col<width; ++col) {
            if(channels == 1) {
                int pv = img.at<uchar>(row, col);
                img.at<uchar>(row,col) = 255 - pv;
            }
            if(channels == 3) {
                Vec3b bgr = img.at<Vec3b>(row, col);
                img.at<Vec3b>(row,col)[0] = 255 - bgr[0];
                img.at<Vec3b>(row,col)[1] = 255 - bgr[1];
                img.at<Vec3b>(row,col)[2] = 255 - bgr[2];
            }
        }
    }
    return img;
}


Mat MainWindow::Erosion(Mat image) {
    auto img = image.clone();
    int width = img.cols;
    int height = img.rows;
    for(int row=0; row<height; ++row) {
        for(int col=0; col<width; ++col) {
            float tmp = 0.0;
            for(int i=-1; i<=1; ++i) {
                for(int j=-1; j<=1; ++j) {
                    if(row+i<0 || row+i>=height || col+j<0 || col+j>=width)
                        continue;
                    if(image.at<uchar>(row+i, col+j) == 255)
                        img.at<uchar>(row,col) = 255;
                }
            }
        }
    }
    return img;
}

Mat MainWindow::Dilation(Mat image) {
    auto img = image.clone();
    int width = img.cols;
    int height = img.rows;
    for(int row=0; row<height; ++row) {
        for(int col=0; col<width; ++col) {
            float tmp = 0.0;
            for(int i=-1; i<=1; ++i) {
                for(int j=-1; j<=1; ++j) {
                    if(row+i<0 || row+i>=height || col+j<0 || col+j>=width)
                        continue;
                    if(image.at<uchar>(row+i, col+j) == 0)
                        img.at<uchar>(row,col) = 0;
                }
            }
        }
    }
    return img;
}

Mat MainWindow::Histogram_Equalization(Mat image) {
    auto img = image.clone();
    int width = img.cols;
    int height = img.rows;
    if(image.channels() == 1) {
        float hist[256] = {0.0};
        for(int row = 0; row < height; row++)
            for(int col = 0; col < width; col++)
                hist[image.at<uchar>(row,col)] += 1.0;
        for(int n=1; n<=255; n++)
            hist[n] += hist[n-1];
        for(int n=0; n<=255; n++)
            hist[n] *= 255.0/(float)(width*height);
        for(int row = 0; row < height; row++)
            for(int col = 0; col < width; col++)
                img.at<uchar>(row,col) = (int)hist[image.at<uchar>(row,col)];

    }
    else if(image.channels() == 3) {
        for(int i=0; i<3; i++) {
            float hist[256] = {0.0};
            for(int row = 0; row < height; row++)
                for(int col = 0; col < width; col++)
                    hist[image.at<Vec3b>(row,col)[i]] += 1.0;
            for(int n=1; n<=255; n++)
                hist[n] += hist[n-1];
            for(int n=0; n<=255; n++)
                hist[n] *= 255.0/(float)(width*height);
            for(int row = 0; row < height; row++)
                for(int col = 0; col < width; col++)
                    img.at<Vec3b>(row,col)[i] = (int)hist[image.at<Vec3b>(row,col)[i]];


        }

    }
    return img;
}


Mat MainWindow::Scaling(Mat image, float scale) {
    int width = image.cols;
    int height = image.rows;
    cv::Mat img;
    int scaled_width = width*scale, scaled_height = scale*height;
    if(image.channels() == 1)
        img = cv::Mat(scaled_height, scaled_width, CV_8UC1, cv::Scalar(0));
    else if(image.channels() == 3)
        img = cv::Mat(scaled_height, scaled_width, CV_8UC3, cv::Scalar(0));
    for(int i=0; i<scaled_height; ++i) {
        for(int j=0; j<scaled_width; ++j) {
            float height_returned, width_returned;
            if((i+1)/scale-1 > 0)
                height_returned = (float)((i+1)/scale-1);
            else
                height_returned = 0.0;
            if((j+1)/scale-1 > 0)
                width_returned = (float)((j+1)/scale-1);
            else
                width_returned = 0.0;
            if(image.channels() == 1) {
                int floor_height = std::floor(height_returned), floor_width = std::floor(width_returned),
                    ceil_height = std::ceil(height_returned), ceil_width = std::ceil(width_returned);
                float upper_rate = (float)ceil_height-height_returned, low_rate=1-upper_rate,
                    left_rate = (float)ceil_width - width_returned, right_rate = 1 - left_rate;
                img.at<uchar>(i,j) = (int)(upper_rate*(left_rate*image.at<uchar>(floor_height,floor_width)+ right_rate*image.at<uchar>(floor_height,ceil_width))
                                            + low_rate*(left_rate*image.at<uchar>(ceil_height,floor_width) + right_rate*image.at<uchar>(ceil_height,ceil_width)));
            }
            else if(image.channels() == 3) {
                int floor_height = std::floor(height_returned), floor_width = std::floor(width_returned),
                    ceil_height = std::ceil(height_returned), ceil_width = std::ceil(width_returned);
                float upper_rate = (float)ceil_height-height_returned, low_rate=1-upper_rate,
                    left_rate = (float)ceil_width - width_returned, right_rate = 1 - left_rate;
                img.at<Vec3b>(i,j) = (Vec3b)(upper_rate*(left_rate*image.at<Vec3b>(floor_height,floor_width)+ right_rate*image.at<Vec3b>(floor_height,ceil_width))
                                            + low_rate*(left_rate*image.at<Vec3b>(ceil_height,floor_width) + right_rate*image.at<Vec3b>(ceil_height,ceil_width)));

            }
        }
    }
    return img;
}


Mat MainWindow::Mean_Filtering(Mat image, int kernal) {
    auto img = image.clone();
    if(kernal == 3) {
        vector<vector<float>> kernel = {{1.0/9.0, 1.0/9.0, 1.0/9.0},
                                        {1.0/9.0, 1.0/9.0, 1.0/9.0},
                                        {1.0/9.0, 1.0/9.0, 1.0/9.0}};
        return colcon(image,kernel);
    }
    else if(kernal == 5) {
        vector<vector<float>> kernel = {{1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0},
                                        {1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0},
                                        {1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0},
                                        {1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0},
                                        {1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0}};
        return colcon(image,kernel);
    }
    else {
        vector<vector<float>> kernel = {{1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0},
                                        {1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0},
                                        {1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0},
                                        {1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0},
                                        {1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0},
                                        {1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0},
                                        {1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0}};
        return colcon(image,kernel);
    }

}

Mat MainWindow::Median_Filtering(Mat image, int kernal) {
    auto img = image.clone();
    int width = img.cols;
    int height = img.rows;
    for(int row = 0; row < height; ++row) {
        for(int col=0; col < width; ++col) {
            if(kernal == 3) {
                vector<int> vec;
                for(int i=-1; i<=1; ++i) {
                    for(int j=-1; j<=1; ++j) {
                        if(row+i < 0 || row+i >=height || col+j < 0 || col+j >=width)
                            continue;
                        vec.push_back(image.at<uchar>(row+i, col+j));
                    }
                }
                std::sort(vec.begin(), vec.end());
                img.at<uchar>(row, col) = vec[(int)(vec.size()/2)];
            }
            else if(kernal == 5) {
                vector<int> vec;
                for(int i=-2; i<=2; ++i) {
                    for(int j=-2; j<=2; ++j) {
                        if(row+i < 0 || row+i >=height || col+j < 0 || col+j >=width)
                            continue;
                        vec.push_back(image.at<uchar>(row+i, col+j));
                    }
                }
                std::sort(vec.begin(), vec.end());
                img.at<uchar>(row, col) = vec[(int)(vec.size()/2)];
            }
            else {
                vector<int> vec;
                for(int i=-3; i<=3; ++i) {
                    for(int j=-3; j<=3; ++j) {
                        if(row+i < 0 || row+i >=height || col+j < 0 || col+j >=width)
                            continue;
                        vec.push_back(image.at<uchar>(row+i, col+j));
                    }
                }
                std::sort(vec.begin(), vec.end());
                img.at<uchar>(row, col) = vec[(int)(vec.size()/2)];
            }
        }
    }
    return img;
}


Mat MainWindow::Gaussian_Filtering(Mat image, int kernal) {
    auto img = image.clone();
    if(kernal == 3) {
        cv::Mat kernel = (cv::Mat_<float>(3,3) <<
            0.05, 0.15, 0.05,
            0.15, 0.20, 0.15,
            0.05, 0.15, 0.05);
        cv::Mat result;
        cv::filter2D(img, result, -1, kernel);
        return result;
    }
    else if(kernal == 5) {
        cv::Mat kernel = (cv::Mat_<float>(5,5) <<
            1,2,4,2,1,
            2,4,16,4,2,
            4,16,64,16,4,
            2,4,16,4,2,
            1,2,4,2,1)/180.0;
        cv::Mat result;
        cv::filter2D(img, result, -1, kernel);
        return result;
    }
    return img;
}


Mat MainWindow::Sobel_Filtering(Mat image, int kernal) {
    auto img = image.clone();
    if(kernal == 0) // X-direction
    {
        vector<vector<float>> kernel = {
            {-1.0, 0.0, 1.0},
            {-2.0, 0.0, 2.0},
            {-1.0, 0.0, 1.0}};
        return colcon(image, kernel);
    }
    else if(kernal == 1)   // Y-direction
    {
        cv::Mat kernel = (cv::Mat_<int>(3,3) <<
                -1,-2,-1,
                0, 0, 0,
                1, 2, 1);
        cv::Mat result;
        cv::filter2D(img, result, -1, kernel);
        return result;
    }
    return img;
}


Mat MainWindow::Laplace_Filtering(Mat image) {
    auto img = image.clone();
    vector<vector<float>> kernel = {
        {0.0, 1.0, 0.0},
        {1.0, -4.0, 1.0},
        {0.0, 1.0, 0.0}};
    return colcon(image, kernel);
}

void MainWindow::getGrandient(Mat img) {
    gradXY = Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
    theta = Mat(img.rows, img.cols, CV_32FC1, cv::Scalar(0));
    for(int i=1; i<img.rows-1; ++i) {
        for(int j=1; j<img.cols-1; ++j) {
            float gradX = float(-img.at<uchar>(i-1,j-1) - 2*img.at<uchar>(i-1,j) - img.at<uchar>(i-1,j+1) + img.at<uchar>(i+1,j-1) + 2*img.at<uchar>(i+1,j) + img.at<uchar>(i+1,j+1));
            float gradY = float(img.at<uchar>(i-1,j+1) + 2*img.at<uchar>(i,j+1) + img.at<uchar>(i+1, j+1) - img.at<uchar>(i-1,j-1) - 2*img.at<uchar>(i,j-1) - img.at<uchar>(i+1,j-1));
            gradXY.at<uchar>(i,j) = sqrt(gradY*gradY + gradX*gradX);
            theta.at<float>(i,j) = atan2(gradY,gradX);
        }
    }
}


Mat MainWindow::nonLocalMaxValue(Mat gradXY, Mat theta) {
    auto img = gradXY.clone();
    for(int i=1; i<gradXY.rows-1; ++i) {
        for(int j=1; j<gradXY.cols-1; ++j) {
            float t = float(theta.at<uchar>(i,j));
            float g = float(gradXY.at<uchar>(i,j));

            if(g== 0.0)
                continue;
            double g0, g1;
            if((t >= -(3 * M_PI/8)) && (t < -(M_PI/8))) {
                g0 = double(gradXY.at<uchar>(i-1,j-1));
                g1 = double(gradXY.at<uchar>(i+1,j+1));
            }
            else if((t >= -(M_PI/8)) && (t < (M_PI/8))) {
                g0 = double(gradXY.at<uchar>(i,j-1));
                g1 = double(gradXY.at<uchar>(i,j+1));
            }
            else if((t >= (M_PI/8)) && (t < (3 * M_PI/8))) {
                g0 = double(gradXY.at<uchar>(i-1,j+1));
                g1 = double(gradXY.at<uchar>(i+1,j-1));
            }
            else {
                g0 = double(gradXY.at<uchar>(i-1,j));
                g1 = double(gradXY.at<uchar>(i+1,j));
            }
            if(g<=g0 || g<=g1) {
                img.at<uchar>(i,j) = 0;
            }
        }
    }
    return img;
}

Mat MainWindow::doubleThreshold(Mat image) {
    auto img = image.clone();
    double minValue, maxValue;
    cv::Point minIdx, maxIdx;

    cv::minMaxLoc(image, &minValue, &maxValue, &minIdx, &maxIdx);
    qDebug() << "最大值" << maxValue;
    for(int i=0; i< img.rows; ++i) {
        for(int j=0; j<img.cols; ++j) {
            if(image.at<uchar>(i,j) >= (int)(maxValue/4))
                img.at<uchar>(i,j)=255;
            else if(image.at<uchar>(i,j) < (int)(maxValue/6))
                img.at<uchar>(i,j) = 0;
            else {
                for(int m=-1; m<=1; ++m) {
                    for(int n=-1; n<=1; ++n) {
                        if(i+m < 0 || i+m >= image.rows || j+n >= image.cols)
                            continue;
                        else {
                            if(image.at<uchar>(i+m,j+n) >= (maxValue/4))
                                img.at<uchar>(i,j) = 255;
                        }
                    }
                }
            }
        }
    }
    return img;
}


void MainWindow::Canny_Filtering(Mat img) {
    Mat tmp = RGB2GRAY(img);
    cv::Mat blurred_image;
    cv::GaussianBlur(tmp, blurred_image, cv::Size(5,5), 0.2);
    getGrandient(blurred_image);
    auto temp_2 = nonLocalMaxValue(gradXY, theta);
    image_result = doubleThreshold(temp_2);
    image_qt_result = cvMat2QImage(image_result, true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);

}

void MainWindow::on_start_clicked() {
    ui->stackedWidget_2->setCurrentIndex(2);
}

void MainWindow::on_image_raw_clicked() {
    ui->stackedWidget_2->setCurrentIndex(0);
}

void MainWindow::on_image_result_clicked() {
    ui->stackedWidget_2->setCurrentIndex(1);
}

void MainWindow::on_image_trans_clicked() {
    ui->stackedWidget->setCurrentIndex(0);
}

void MainWindow::on_image_filter_clicked() {
    ui->stackedWidget->setCurrentIndex(1);
}

void MainWindow::on_save_result_clicked() {
    QString fileName = QFileDialog::getSaveFileName(this,
                                                    tr("保存图片"),
                                                    QDir::homePath() + "/untitled.png",
                                                    tr("Image Files (*.png *.xpm *.jpg"));
    if(!fileName.isEmpty()) {
        bool saved = image_qt_result.save(fileName);
        if(saved) {
            QMessageBox::information(nullptr, tr("Save Image"), tr("Image saved succcessfully."));
        } else {
            QMessageBox::warning(nullptr, tr("Save Image"), tr("Failed to save image."));
        }
    }
}

// 转为灰度图 按键
void MainWindow::on_gray_clicked() {
    image_result = RGB2GRAY(image_raw);
    image_qt_result = cvMat2QImage(image_result, false);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

// 转为HSV 按键
void MainWindow::on_hsv_clicked() {
    image_result = RGB2HSV(image_raw);
    image_qt_result = cvMat2QImage(image_result, false);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

// 水平镜像 按键
void MainWindow::on_horizon_clicked() {
    image_result = Horizontal_Mirroring(image_raw);
    image_qt_result = cvMat2QImage(image_result, true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

// 垂直镜像 按键
void MainWindow::on_vertical_clicked() {
    image_result = Vertical_Mirroring(image_raw);
    image_qt_result = cvMat2QImage(image_result, true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}


void MainWindow::on_rotate_slider_valueChanged(int value) {
    image_result = Rotate(image_raw, (float)value);
    image_qt_result = cvMat2QImage(image_result, true);

    if(image_qt_result.width() > 750 || image_qt_result.height() > 520) {
        float reduce_scale_;
        if((float)((float)image_qt_result.width()/750.0)>=(float)((float)image_qt_result.height()/520.0)) {
            reduce_scale_ = 750/(float)image_qt_result.width();
        }
        else
            reduce_scale_ = 520/(float)image_qt_result.height();

        image_qt_reduced = image_qt_result.scaled((int)(image_qt_result.width() * reduce_scale_ -1),
                                                  (int)(image_qt_result.height()*reduce_scale_ -1),
                                                  Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui->label->setPixmap(QPixmap::fromImage(image_qt_reduced));
        ui->stackedWidget_2->setCurrentIndex(1);
    }
    else {
        ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
        ui->stackedWidget_2->setCurrentIndex(1);
    }
}

// 默认阈值分割 按键
void MainWindow::on_default_segmentation_clicked() {
    auto tmp = RGB2GRAY(image_raw);
    image_result = Threshold_Segmentation(tmp, 127);
    image_qt_result = cvMat2QImage(image_result, true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

// 彩色阈值分割 按键
void MainWindow::on_color_segmentation_clicked() {
    image_result = Threshold_Segmentation(image_raw, 127);
    image_qt_result = cvMat2QImage(image_result, true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}


void MainWindow::on_dajin_clicked() {
    int max_index = dajin(image_raw);
    auto temp = RGB2GRAY(image_raw.clone());
    auto tmp = Threshold_Segmentation(temp, max_index);
    image_qt_result = cvMat2QImage(tmp, true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);

}

void MainWindow::on_reverse_2_clicked() {
    image_result = Reverse(image_raw);
    image_qt_result = cvMat2QImage(image_result, true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);

}


void MainWindow::on_reverse_gray_clicked() {
    auto tmp = RGB2GRAY(image_raw);
    image_result = Reverse(tmp);
    image_qt_result = cvMat2QImage(image_result, true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);

}

void MainWindow::on_Erosion_2_clicked() {
    auto temp = RGB2GRAY(image_raw);
    auto tmp = Threshold_Segmentation(temp,127);
    image_result = Erosion(tmp);
    image_qt_result = cvMat2QImage(image_result, true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

void MainWindow::on_Dilation_2_clicked() {
    auto temp = RGB2GRAY(image_raw);
    auto tmp = Threshold_Segmentation(temp,127);
    image_result = Dilation(tmp);
    image_qt_result = cvMat2QImage(image_result, true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

void MainWindow::on_open_clicked() {
    auto temp = RGB2GRAY(image_raw);
    auto tmp = Threshold_Segmentation(temp,127);
    image_result = Erosion(tmp);
    image_result = Dilation(image_result);
    image_qt_result = cvMat2QImage(image_result, true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}


void MainWindow::on_close_clicked() {
    auto temp = RGB2GRAY(image_raw);
    auto tmp = Threshold_Segmentation(temp,127);
    image_result = Dilation(tmp);
    image_result = Erosion(image_result);
    image_qt_result = cvMat2QImage(image_result, true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}


void MainWindow::on_histograam_clicked() {
    auto tmp = RGB2GRAY(image_raw);
    image_result = Histogram_Equalization(tmp);
    image_qt_result = cvMat2QImage(image_result, true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

void MainWindow::on_hist_color_clicked() {
    image_result = Histogram_Equalization(image_raw);
    image_qt_result = cvMat2QImage(image_result, true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

void MainWindow::on_horizontalSlider_valueChanged(int value) {
    auto tmp = RGB2GRAY(image_raw);
    tmp = Scaling(image_raw, (float)value/100);
    image_qt_result = cvMat2QImage(tmp,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}


void MainWindow::on_mean_3_clicked() {
    auto tmp = RGB2GRAY(image_raw);
    tmp = Mean_Filtering(tmp,3);
    image_qt_result = cvMat2QImage(tmp,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

void MainWindow::on_mean_5_clicked() {
    auto tmp = RGB2GRAY(image_raw);
    tmp = Mean_Filtering(tmp,5);
    image_qt_result = cvMat2QImage(tmp,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

void MainWindow::on_mean_7_clicked() {
    auto tmp = RGB2GRAY(image_raw);
    tmp = Mean_Filtering(tmp,7);
    image_qt_result = cvMat2QImage(tmp,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}



void MainWindow::on_medium_3_clicked() {
    auto tmp = RGB2GRAY(image_raw);
    tmp = Median_Filtering(tmp,3);
    image_qt_result = cvMat2QImage(tmp,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

void MainWindow::on_medium_5_clicked() {
    auto tmp = RGB2GRAY(image_raw);
    tmp = Median_Filtering(tmp,5);
    image_qt_result = cvMat2QImage(tmp,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

void MainWindow::on_medium_7_clicked() {
    auto tmp = RGB2GRAY(image_raw);
    tmp = Median_Filtering(tmp,7);
    image_qt_result = cvMat2QImage(tmp,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}


void MainWindow::on_Gaussian_3_clicked() {
    auto tmp = RGB2GRAY(image_raw);
    tmp = Gaussian_Filtering(tmp,3);
    image_qt_result = cvMat2QImage(tmp,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

void MainWindow::on_Gaussian_5_clicked() {
    auto tmp = RGB2GRAY(image_raw);
    tmp = Gaussian_Filtering(tmp,5);
    image_qt_result = cvMat2QImage(tmp,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

void MainWindow::on_sobel_x_clicked() {
    auto tmp = RGB2GRAY(image_raw);
    tmp = Sobel_Filtering(tmp,0);
    image_qt_result = cvMat2QImage(tmp,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

void MainWindow::on_sobel_y_clicked() {
    auto tmp = RGB2GRAY(image_raw);
    tmp = Sobel_Filtering(tmp,1);
    image_qt_result = cvMat2QImage(tmp,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

void MainWindow::on_sobel_abs_clicked() {
    auto tmp = RGB2GRAY(image_raw);
    tmp = Sobel_Filtering(tmp,0) + Sobel_Filtering(tmp,1);
    image_qt_result = cvMat2QImage(tmp,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

void MainWindow::on_laplace_1_clicked() {
    auto tmp = RGB2GRAY(image_raw);
    tmp = Laplace_Filtering(tmp);
    image_qt_result = cvMat2QImage(tmp,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

void MainWindow::on_laplace_reduce_clicked() {
    auto tmp = RGB2GRAY(image_raw);
    tmp = tmp - Laplace_Filtering(tmp);
    image_qt_result = cvMat2QImage(tmp,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

void MainWindow::on_laplace_edge_clicked() {
    auto tmp = RGB2GRAY(image_raw);
    image_result = Laplace_Filtering(tmp);
    double minValue, maxValue;
    cv::Point minIdx, maxIdx;
    cv::minMaxLoc(image_result, &minValue, &maxValue, &minIdx, &maxIdx);
    image_result = Threshold_Segmentation(image_result, (int)(maxValue/4));
    image_qt_result = cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}


void MainWindow::on_canny_1_clicked() {
    Canny_Filtering(image_raw);
}

void MainWindow::on_noise_clicked() {
    image_raw = addSaltNoise(image_raw, 3000);
    image_qt_result = cvMat2QImage(image_raw,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget_2->setCurrentIndex(1);
}

void MainWindow::on_opencv_mode_clicked() {
    process_mode = true;
    qDebug() << "true";
}

void MainWindow::on_raw_mode_clicked() {
    process_mode = false;
    qDebug() << "false";
}
