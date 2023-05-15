// Base includes
#include <iostream>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <math.h>

// Namespaces for simplicity
using namespace cv;
using namespace std;

// Emotion model ferplus
dnn::Net net = dnn::readNetFromONNX("../emotion-ferplus-8.onnx");

// soft max function

void softmax(Mat &v){
    exp(v,v);
    auto s = sum(v)[0];
    v/=s;
}

int main(int argc, char* argv[]){

    double scale = 2.0;

    CascadeClassifier faceCascade;
    faceCascade.load("/usr/local/Cellar/opencv/4.7.0_4/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml");

    // Mat image = imread("../face.jpg", IMREAD_COLOR);
    VideoCapture cap(0);
    if(!cap.isOpened()){
        return -1;
    }


    for (;;){

        Mat image;
        cap >> image;
        if(image.empty()){
            cout << "Image not loading" << endl;
            return -1;
        }

        Mat grayScale;
        cvtColor(image, grayScale, COLOR_BGR2GRAY);
        resize(grayScale, grayScale, Size(grayScale.size().width / scale, grayScale.size().height / scale));

        vector<Rect> faces;

        faceCascade.detectMultiScale(grayScale, faces, 1.1,3,0, Size(10,10));

        for(Rect area : faces){

            Scalar drawColor = Scalar(255,0,0);
            Point center(cvRound((area.x + area.width*0.5) * scale), cvRound((area.y + area.height*0.5) * scale));
            int radius = cvRound((area.width + area.height)*0.25 * scale);
            circle(image, center, radius, drawColor, 3, 8, 0);

            // Image preprocessing for FER+ model
            Mat face = grayScale(area);
            resize(face,face,Size(64,64));
            face.convertTo(face,CV_32F);

            Mat blob = dnn::blobFromImage(face, 1, Size(64,64), Scalar(0),false,false, CV_32F);
            net.setInput(blob);

            Mat result = net.forward();
            softmax(result);

            cout << "Soft max result: " << result << endl;

            Point maxLoc;
            minMaxLoc(result,nullptr,nullptr,nullptr,&maxLoc);

            // Vector set of emotions
            vector<std::string> emotions = {"neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"};

            string emotion = emotions[maxLoc.x];

            // Draw the emotion on the image
            putText(image, emotion, center, cv::FONT_HERSHEY_SIMPLEX, 1, drawColor, 2);



            // Mat blob = dnn::blobFromImage
        }

        // namedWindow("Display WIndow", WINDOW_AUTOSIZE);
        imshow("Display Image", image);

        if(waitKey(30) >= 0){
            break;
        }


    }



    return 0;
    
}