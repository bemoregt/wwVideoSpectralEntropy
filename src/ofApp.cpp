#include "ofApp.h"

using namespace cv;
using namespace ofxCv;

void fftshift(Mat & in, Mat & out);
void synthesizeFilterH(Mat& inputOutput_H, cv::Point center, int radius);
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);

//--------------------------------------------------------------
void ofApp::setup(){
    
    vid1.load("/Users/mun/Desktop/goose.mov");

    vid1.play();
    //vid1.initGrabber(384, 384);
    
    ofBackground(50);
}

//--------------------------------------------------------------
void ofApp::update(){

    vid1.update();
    if(vid1.isFrameNew()) {
        cap.setFromPixels(vid1.getPixels());
        cap.update();
        
        cv::Mat img = toCv(cap);
        cv::Mat img2, img3;
        cv::cvtColor(img, img2, CV_RGB2GRAY);
        // resize
        float ratio = 256.0 / img2.cols;
        cv::resize(img2, img3, cv::Size(img2.cols*ratio, img2.rows*ratio));
        
        // 2d DFT
        cv::Mat planes[] = {cv::Mat_<float>(img3), cv::Mat::zeros(img3.size(), CV_32F)};
        cv::Mat complexImg;
        cv::merge(planes, 2, complexImg);
        cv::dft(complexImg, complexImg);
        // added 190709
        fftshift(complexImg, complexImg);
        
        cv::split(complexImg, planes);
        
        cv::Mat mag, logmag;
        cv::Mat mag1;
        
        cv::magnitude(planes[0], planes[1], mag);
        //ofLog() << mag.rows << mag.cols << endl;
        
        cv::log(mag, logmag);
        
        cv::normalize(logmag, mag1, 255, 0, cv::NORM_MINMAX, CV_8U);
        //
        for(int i=125; i<131; i++){
            for(int j=125; j<131; j++){
                mag1.data[i * 256 + j]=0;
            }
        }//for
        toOf(mag1, spectrum);
        spectrum.update();
        
        gg.setFromPixels(spectrum.getPixels().getChannel(1));
        
        histogramG = histogram.getHistogram(gg, 256);
        
        gentropy = 0.0;
        
        // Entropy --------------------------------------
        for (int i = 0; i < histogramG.size(); ++i)
        {
            gentropy += histogramG.data()[i]*log2(histogramG.data()[i]+1);
        }
        //gentropy /= histogramG.size();
        gentropy= -gentropy;
    }//if
}

//--------------------------------------------------------------
void ofApp::draw(){

    ofSetColor(255);
    vid1.draw(0, 0, 400, 400);
    spectrum.draw(400, 0, 400, 400);
    
    ofSetColor(255, 255, 0);
    ofDrawBitmapString("Frame/sec= " + ofToString(ofGetFrameRate()), 20, 20);
    ofSetColor(255, 255, 0);
    ofDrawBitmapString("Spectral Entropy= " + ofToString(gentropy), 20, 40);
    
    ofSetColor(0, 255, 0);
    //gg.draw(256, 0);
    drawHistogram(histogramG);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
//--------------------------------------------------
void fftshift(Mat & in, Mat & out) {
    out = in.clone();
    int mx1, my1, mx2, my2;
    mx1 = out.cols / 2;
    my1 = out.rows / 2;
    mx2 = int(ceil(out.cols / 2.0));
    my2 = int(ceil(out.rows / 2.0));
    Mat q0(out, cv::Rect(0, 0, mx2, my2));
    Mat q1(out, cv::Rect(mx2, 0, mx1, my2));
    Mat q2(out, cv::Rect(0, my2, mx2, my1));
    Mat q3(out, cv::Rect(mx2, my2, mx1, my1));
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q2.copyTo(tmp);
    q1.copyTo(q2);
    tmp.copyTo(q1);
    vconcat(q1, q3, out);
    vconcat(q0, q2, tmp);
    hconcat(tmp, out, out);
}

//--------------------------------------------------------------
void ofApp::drawHistogram(vector<float> & h) {
    ofBeginShape();
    ofNoFill();
    ofSetLineWidth(3);
    for (int i=0; i<h.size(); i++) {
        float x = ofMap(i, 0, h.size(), 0, 320);
        float y = ofMap(h[i], 0, 0.3, 240, 0);
        ofVertex(x, y);
    }
    ofEndShape();
}
//-----------------------------------------------------------------------
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
    Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI, DFT_SCALE);
    Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
    Mat complexH;
    merge(planesH, 2, complexH);
    Mat complexIH;
    mulSpectrums(complexI, complexH, complexIH, 0);
    idft(complexIH, complexIH);
    split(complexIH, planes);
    outputImg = planes[0];
}
// ----------------------------------------------------------------------------
void synthesizeFilterH(Mat& inputOutput_H, cv::Point center, int radius)
{
    cv::Point c2 = center, c3 = center, c4 = center;
    c2.y = inputOutput_H.rows - center.y;
    c3.x = inputOutput_H.cols - center.x;
    c4 = cv::Point(c3.x,c2.y);
    circle(inputOutput_H, center, radius, 0, -1, 8);
    circle(inputOutput_H, c2, radius, 0, -1, 8);
    circle(inputOutput_H, c3, radius, 0, -1, 8);
    circle(inputOutput_H, c4, radius, 0, -1, 8);
}
