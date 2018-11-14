// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "time.h"
#include <fstream>

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = MAX_PATH - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0.4;
		int pH = 50;
		int pL = k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the default camera (i.e. the built in web cam)
	if (!cap.isOpened()) // opening the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

std::vector<Point2f> openAndReadFile() {
	FILE* f = fopen("files/points3.txt", "r");
	int pointCount = 0;
	fscanf(f, "%d", &pointCount);

	std::vector<Point2f> points(pointCount);

	for (int i = 0; i < pointCount; i++) {
		float x, y;
		fscanf(f, "%f%f", &x, &y);
		Point2f p;
		p.x = x;
		p.y = y;
		points.at(i) = p;
	}
	fclose(f);
	return points;
}

Point2f calculateThetas(std::vector<Point2f> points) {
	int n = points.size();

	float sumxy = 0;
	float sumx = 0;
	float sumy = 0;
	float sumxx = 0;
	for (int i = 0; i < n; i++) {
		Point2f p = points.at(i);
		sumxy += p.x * p.y;
		sumx += p.x;
		sumy += p.y;
		sumxx += p.x * p.x;
	}
	float theta1 = ((n * sumxy) - sumx * sumy) / (n * sumxx - sumx * sumx);
	float theta0 = (1.0f / n) * (sumy - theta1 * sumx);
	return Point2f(theta0, theta1);
}

Point2f calculateBetaRo(std::vector<Point2f> points) {
	int n = points.size();

	float sumxminusy = 0;
	float sumxy = 0;
	float sumx = 0;
	float sumy = 0;
	for (int i = 0; i < n; i++) {
		Point2f p = points.at(i);
		sumxy += p.x * p.y;
		sumxminusy += p.y * p.y - p.x * p.x;
		sumx += p.x;
		sumy += p.y;
	}
	float beta = -atan2(2 * sumxy - (2 * sumx * sumy) / n,
		sumxminusy + (sumx * sumx) / n - (sumy * sumy) / n) / 2;
	float ro = (cos(beta) * sumx + sin(beta) * sumy) / n;
	return Point2f(beta, ro);
}

void displayPoints() {

	std::vector<Point2f> points = openAndReadFile();

	Mat img(500, 500, CV_8UC3);

	for (int i = 0; i < points.size(); i++) {
		Point2f point = points.at(i);
		if (point.x > 0 && point.x < 500 && point.y > 0 && point.y < 500) {
			cv::drawMarker(img, cv::Point(point.x, point.y), cv::Scalar(0, 0, 255), MARKER_CROSS, 10, 1);
			img.at<Vec3b>(point.y, point.x)[0] = 255; //blue
			img.at<Vec3b>(point.y, point.x)[1] = 10; //green
			img.at<Vec3b>(point.y, point.x)[2] = 10; //red
		}
	}

	Point2f thetas = calculateThetas(points);

	std::cout << thetas.x << " " << thetas.y;

	if (thetas.y > 1) {
		int x1 = -thetas.x / thetas.y;
		int x2 = (500 - thetas.x) / thetas.y;

		line(img, Point(x1, 0), Point(x2, 500), Scalar(0, 255, 0));
	}
	else {
		int y1 = thetas.x + thetas.y * 0;
		int y2 = thetas.x + thetas.y * 500;

		line(img, Point(0, y1), Point(500, y2), Scalar(0, 255, 0));
	}


	Point2f betaro = calculateBetaRo(points);
	int y1, y2;
	if (betaro.x < 0.1) {
		y1 = betaro.y / cos(betaro.x);
		y2 = (betaro.y - 500 * sin(betaro.x)) / cos(betaro.x);
		line(img, Point(y1, 0), Point(y2, 500), Scalar(255, 0, 0));
	}
	else {
		y1 = betaro.y / sin(betaro.x);
		y2 = (betaro.y - 500 * cos(betaro.x)) / sin(betaro.x);
		line(img, Point(0, y1), Point(500, y2), Scalar(255, 0, 0));
	}

	std::cout << "\n" << betaro.x << " " << betaro.y;
	imshow("Points", img);
	waitKey();
}

float fdex(float x, float theta0, float theta1) {
	return theta0 + theta1 * x;
}

float jacobi(std::vector<Point2f> points, float theta1, float theta2) {
	float jacobi = 0;
	for (int i = 0; i < points.size(); i++) {
		float elem = fdex(points.at(i).x, theta1, theta2) - points.at(i).y;
		jacobi += elem * elem;
	}

	return jacobi / 2;
}

Point2f jacobiderivatives(std::vector<Point2f> points, float theta1, float theta2) {
	float theta0Deriv = 0;
	float theta1Deriv = 0;
	for (int i = 0; i < points.size(); i++) {
		theta0Deriv += fdex(points.at(i).x, theta1, theta2) - points.at(i).y;
		theta1Deriv += (fdex(points.at(i).x, theta1, theta2) - points.at(i).y) * points.at(i).x;
	}

	return Point2f(theta0Deriv, theta1Deriv);
}

void gradientDescent() {

	std::vector<Point2f> points = openAndReadFile();

	Mat img(500, 500, CV_8UC3);

	for (int i = 0; i < points.size(); i++) {
		Point2f point = points.at(i);
		if (point.x > 0 && point.x < 500 && point.y > 0 && point.y < 500) {
			cv::drawMarker(img, cv::Point(point.x, point.y), cv::Scalar(0, 0, 255), MARKER_CROSS, 10, 1);
			img.at<Vec3b>(point.y, point.x)[0] = 255; //blue
			img.at<Vec3b>(point.y, point.x)[1] = 10; //green
			img.at<Vec3b>(point.y, point.x)[2] = 10; //red
		}
	}
	float error = 1;
	float previous_jacobi = 0;
	float theta1 = 0;
	float theta0 = 0;
	float alfa = 0.0000001;
	while (abs(error) > 0.01) {

		Point2f newThetas = jacobiderivatives(points, theta0, theta1);

		error = previous_jacobi - jacobi(points, newThetas.x, newThetas.y);

		theta0 = theta0 - alfa * newThetas.x;
		theta1 = theta1 - alfa * newThetas.y;

		printf("new theta 0 %f\n", theta0);
		printf("new theta 1 %f\n", theta1);

		int y1 = theta0 + theta1 * 0;

		int y2 = theta0 + theta1 * 500;

		line(img, Point2f(0, y1), Point2f(500, y2), Scalar(0, 255, 0));
		imshow("Points", img);
		waitKey();
	}

}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

float calculateDistance(float a, float b, float c, Point2i point) {
	return fabs(a * point.x + b * point.y + c) / sqrt(a*a + b*b);
}

void ransac()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat colorsrc = imread(fname, CV_LOAD_IMAGE_COLOR);


		std::vector<Point2i> blackPoints;

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				uchar pxl = src.at<uchar>(i, j);
				if (pxl == 0) {
					blackPoints.push_back(Point2i(j, i));
				}
			}
		}

		float t = 10;
		float q = 0.3f;
		float p = 0.99f;

		int N = log(1 - p) / log(1 - pow(q, 2));

		srand(time(NULL));
		
		std::vector<int> selectedPoints;
		float a, b, c;
		int consensus_set_cardinal = 0;
		int ip1 = 0, ip2 = 0;

		// Ransac
		for (int i = 0; i < N; i++) {

			int nip1 = 0;
			int nip2 = 0;
			// pick 2 points randomly

			while (nip1 == nip2) {
				nip1 = rand() % blackPoints.size();
				nip2 = rand() % blackPoints.size();
			}

			//calculate the line parameters
			float na = blackPoints.at(nip1).y - blackPoints.at(nip2).y;
			float nb = blackPoints.at(nip2).x - blackPoints.at(nip1).x;
			float nc = blackPoints.at(nip1).x * blackPoints.at(nip2).y - blackPoints.at(nip2).x * blackPoints.at(nip1).y;

			float nconsensus_set_cardinal = 0;

			// compute consensus set
			for (int j = 0; j < blackPoints.size(); j++) {
				float distance = calculateDistance(na, nb, nc, blackPoints.at(j));

				if (distance <= t) {
					nconsensus_set_cardinal++;
				}
			}

			line(colorsrc, blackPoints.at(nip1), blackPoints.at(nip2), Scalar(0, 255, 0));

			//persist if the consensus set is max so far
			if (nconsensus_set_cardinal > consensus_set_cardinal) {
				consensus_set_cardinal = nconsensus_set_cardinal;
				a = na;
				b = nb;
				c = nc;
				ip1 = nip1;
				ip2 = nip2;
			}

			if (consensus_set_cardinal > q * blackPoints.size()) {
				break;
			}

		}
		line(colorsrc, blackPoints.at(ip1), blackPoints.at(ip2), Scalar(255, 0, 0));
		//show the image
		imshow("My Window", colorsrc);

		// Wait until user press some key
		waitKey(0);
	}
}

struct peak {
	int theta, ro, hval;
	bool operator < (const peak& o) const {
		return hval > o.hval;
	}
};

void haugh() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat colorImg = imread(fname, CV_LOAD_IMAGE_COLOR);

		std::vector<Point2i> whitePoints;

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				uchar pxl = src.at<uchar>(i, j);
				if (pxl == 255) {
					whitePoints.push_back(Point2i(j, i));
				}
			}
		}

		int D = sqrt(pow(src.rows, 2) + pow(src.cols, 2));

		Mat Hough(360, D + 1, CV_32SC1); //matrix with int values

		for (int i = 0; i < Hough.rows; i++) {
			for (int j = 0; j < Hough.cols; j++) {
				Hough.at<int>(i, j) = 0;;
			}
		}

		// cycle through the points in the image
		for (int k = 0; k < whitePoints.size(); k++) {
			// try out all the lines passing through that point
			for (int theta = 0; theta < 360; theta++) {
				if (theta < 180 || theta > 270) {
					float rad = theta * PI / 180;
					int ro = whitePoints.at(k).x * cos(rad) + whitePoints.at(k).y * sin(rad);
					if (ro >= 0)
						Hough.at<int>(theta, ro)++;
				}
			}
		}

		int maxHough = 0;


		for (int i = 0; i < Hough.rows; i++) {
			for (int j = 0; j < Hough.cols; j++) {
				if (Hough.at<int>(i, j) > maxHough) {
					maxHough = Hough.at<int>(i, j);
				}
			}
		}
		Mat houghImg;
		Hough.convertTo(houghImg, CV_8UC1, 255.f / maxHough);

		std::vector<peak> peaks;

		for (int i = 0; i < Hough.rows; i++) {
			for (int j = 0; j < Hough.cols; j++) {
				//parse the points with the 7x7 window
				int localMax = 0;
				for (int k = i - 3; k < i + 4; k++) {
					int localTheta = (k + 360) % 360;
					for (int q = j - 3; q < j + 4; q++) {
						if (q < 0 || q >= D + 1) {
							continue;
						}

						if (Hough.at<int>(localTheta, q) > localMax) {
							localMax = Hough.at<int>(localTheta, q);
						}
					}
				}
				if (Hough.at<int>(i, j) == localMax) {
					peaks.push_back(peak{ i, j, localMax });
				}

			}
		}

		std::sort(peaks.begin(), peaks.end());

		for (int i = 0; i < 9; i++) {
			int theta = peaks.at(i).theta;
			int ro = peaks.at(i).ro;
			float rad = theta * PI / 180;
			int y1, y2;
			if (rad < 0.1) {
				y1 = ro / cos(rad);
				y2 = (ro - 102 * sin(rad)) / cos(rad);
				line(colorImg, Point(y1, 0), Point(y2, 102), Scalar(0, 255, 0), 2);
			}
			else {
				y1 = ro / sin(rad);
				y2 = (ro - 102 * cos(rad)) / sin(rad);
				line(colorImg, Point(0, y1), Point(102, y2), Scalar(255, 0, 0), 2);
			}
		}

		imshow("orig img", src);
		imshow("color img", colorImg);
		imshow("hough space", houghImg);

		// Wait until user press some key
		waitKey(0);
	}
}

int calcChamDist(int i, int j, int y, int x) {
	int weight = 0;
	if (abs(i - y) == 1) {
		//not on same row
		if (abs(j - x) == 1) {
			//diagonal case
			weight = 3;
		}
		else {
			//vertical distance
			weight = 2;
		}
	}
	else {
		if (abs(j - x) == 1) {
			//horizontal distance
			weight = 2;
		}
		else {
			//same distance
			weight = 0;
		}
	}
	return weight;
}


Mat ChamferDistance() {
	Mat src;
	Mat cham_distance;
	// Read image from file 
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		src.copyTo(cham_distance);
		
		//first iteration
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				float min = 255;
				for (int row = i - 1; row < i + 1; row++) {
					for (int col = j - 1; col < j + 2; col++) {
						if (row < 0 || col < 0 || col > src.cols - 1) {
							continue;
						}
						float dist = cham_distance.at<uchar>(row, col) + calcChamDist(i,j, row, col);
						if (min > dist) {
							min = dist;
						}
					}
				}
				cham_distance.at<uchar>(i, j) = min;
			}
		}

		Mat cham_distance_first_iteration;
		cham_distance.copyTo(cham_distance_first_iteration);

		//second iteration
		for (int i = src.rows - 1; i > 0; i--) {
			for (int j = src.cols - 1; j > 0; j--) {
				float min = 255;
				for (int row = i + 1; row > i - 1; row--) {
					for (int col = j + 1; col > j - 2; col--) {
						if (row > src.rows - 1 || col > src.cols - 1 || col < 0) {
							continue;
						}
						float dist = cham_distance.at<uchar>(row, col) + calcChamDist(i, j, row, col);
						if (min > dist) {
							min = dist;
						}
					}
				}
				cham_distance.at<uchar>(i, j) = min;
			}
		}

		imshow("orig img", src);
		imshow("cham_dist_first_iter", cham_distance_first_iteration);
		imshow("cham_dist", cham_distance);
		
		// Wait until user press some key
		waitKey(0);
	}
	return cham_distance;
}

void CalculateChamDifference() {
	Mat template_img;
	Mat unknown_img;
	// Read image from file 
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		template_img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	}

	if(openFileDlg(fname))
	{
		unknown_img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	}

	Mat unknown_img_cham_dist = ChamferDistance();

	int totalDifference = 0;
	int borderPixelCount = 0;
	//perform comparison 
	for (int i = 0; i < template_img.rows; i++) {
		for (int j = 0; j < template_img.cols; j++) {
			if (template_img.at<uchar>(i, j) == 0) {
				//border pixel
				totalDifference += unknown_img_cham_dist.at<uchar>(i, j);
				borderPixelCount++;
			}
		}
	}
	
	std::cout << "Difference is " << ((borderPixelCount == 0) ? 0 : totalDifference / borderPixelCount);

	imshow("orig img", template_img);
	imshow("unknown_img", unknown_img);

	// Wait until user press some key
	waitKey(0);
}

float meanOfProperty(Mat properties, int index) {
	float sum = 0;
	for (int i = 0; i < properties.rows; i++) {
		sum += properties.at<uchar>(i, index);
	}
	return sum / properties.rows;
}

void saveMatToCsv(Mat &matrix, std::string filename) {
	std::ofstream outputFile(filename);
	outputFile << format(matrix, cv::Formatter::FMT_CSV) << std::endl;
	outputFile.close();
}

void statisticalAnalisys() {
	char folder[256] = "files/faces";
	char fname[256];
	
	int N = 361;
	int P = 400;

	Mat properties = Mat(P, N, CV_8UC1);

	//read the images into the properties mat
	for (int i = 1; i <= P; i++) {
		sprintf(fname, "%s/face%05d.bmp", folder, i);
		Mat img = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int counter = 0;
		for (int k = 0; k < img.rows; k++) {
			for (int l = 0; l < img.cols; l++) {
				properties.at<uchar>(i - 1, counter) = img.at<uchar>(k, l);
				counter++;
			}
		}
	}

	//calculate mean of features
	std::vector<float> means(400);
	
	for (int i = 0; i < N; i++) {
		float sum = 0;
		for (int j = 0; j < P; j++) {
			sum += properties.at<uchar>(j, i);
		}
		means.at(i) = sum / P;
	}

	//calculate the covariance matrix
	Mat cov = Mat(N, N, CV_32FC1);

	for (int i = 0; i < N ; i++) {
		for (int j = 0; j < N ; j++) {
			float sum = 0;

			for (int k = 0; k < P; k++) {
				sum += (properties.at<uchar>(k, i) - means.at(i)) * (properties.at<uchar>(k, j) - means.at(j));
			}
			cov.at<float>(i, j) = sum / P;
		}
	}

	saveMatToCsv(cov, "covariance.csv");

	//calculate the correlation matrix
	Mat cor = Mat(N, N, CV_32FC1);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cor.at<float>(i, j) = cov.at<float>(i,j) / (sqrt(cov.at<float>(i,i)) * sqrt(cov.at<float>(j, j)));
		}
	}

	saveMatToCsv(cor, "corelation.csv");


	Mat chart1 = Mat(256, 256, CV_8UC1, 255);
	Mat chart2 = Mat(256, 256, CV_8UC1, 255);
	Mat chart3 = Mat(256, 256, CV_8UC1, 255);
	
	for (int i = 0; i < P; i++) {
		chart1.at<uchar>(properties.at<uchar>(i, 5 * 19 + 4), properties.at<uchar>(i, 5 * 19 + 14)) = 0;
		chart2.at<uchar>(properties.at<uchar>(i, 10 * 19 + 3), properties.at<uchar>(i, 9 * 19 + 15)) = 0;
		chart3.at<uchar>(properties.at<uchar>(i, 5 * 19 + 4), properties.at<uchar>(i, 18 * 19)) = 0;
	}

	imshow("chart1", chart1);
	imshow("chart2", chart2);
	imshow("chart3", chart3);
	waitKey(0);
}

float squaredeEuclidDist(Point3i first, Point3i second) {
	return abs(pow(first.x - second.x,2)) + abs(pow(first.y - second.y,2));
}

void KMeans() {

	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat output = Mat(src.rows, src.cols, CV_8UC3, Vec3b(255,255,255));
		Mat voronoi = Mat(src.rows, src.cols, CV_8UC3, Vec3b(255, 255, 255));

		//find all the forms 
		std::vector<Point3i> blackPoints;

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				uchar pxl = src.at<uchar>(i, j);
				if (pxl == 0) {
					blackPoints.push_back(Point3i(j, i, -1));
				}
			}
		}

		int n = blackPoints.size();


		srand(time(NULL));

		//find randomly K points
		int K = 5;

		std::vector<Point3i> centers;

		int nrOfCenters = 0;
		while (nrOfCenters < K) {

			int index = abs(rand() % n);

			bool goodIndex = true;
			for (int i = 0; i < centers.size(); i++) {
				if (centers.at(i).x == blackPoints.at(index).x &&
					centers.at(i).y == blackPoints.at(index).y) {
					goodIndex = false;
					break;
				}
			}

			if (goodIndex) {
				blackPoints.at(index).z = nrOfCenters;
				nrOfCenters++;
				centers.push_back(blackPoints.at(index));
			}
		}

		//perform step 2 and 3
		bool continueExecution = true;
		int executionCycles = 0;

		std::vector<Point3i> previousCenters;
		previousCenters = centers;

		while (continueExecution) {

			continueExecution = false;

			//step 2 iterate through all the forms
			for (int i = 0; i < n; i++) {
				//check where they belong
				float dist = 1000000;
				for (int k = 0; k < K; k++) {
					float newdist = squaredeEuclidDist(blackPoints.at(i), centers.at(k));
					if (dist > newdist) {
						dist = newdist;
						blackPoints.at(i).z = centers.at(k).z;
					}
				}
			}

			//update clusters
			for (int k = 0; k < K; k++) {

				int nrOfFormsInGroup = 0;
				int sumx = 0;
				int sumy = 0;

				for (int i = 0; i < n; i++) {
					if (blackPoints.at(i).z == k) {
						sumx += blackPoints.at(i).x;
						sumy += blackPoints.at(i).y;
						nrOfFormsInGroup++;
					}
				}

				centers.at(k).x = sumx / nrOfFormsInGroup;
				centers.at(k).y = sumy / nrOfFormsInGroup;
			}
			executionCycles++;

			//check if there are changes in centers locations
			for (int i = 0; i < K; i++) {
				if (centers.at(i).x != previousCenters.at(i).x ||
					centers.at(i).y != previousCenters.at(i).y) {
					continueExecution = true;
					break;
				}
			}

			previousCenters = centers;

		}

		std::vector<Vec3b> colors(K);
		for (int i = 0; i < K; i++)
			colors.at(i) = Vec3b(rand() % 256, rand() % 256,  rand() % 256);
	
		for (int i = 0; i < n; i++) {
			output.at<Vec3b>(blackPoints.at(i).y, blackPoints.at(i).x) = colors.at(blackPoints.at(i).z);
		}

		//voronoi
		for (int i = 0; i < output.rows; i++) {
			for (int j = 0; j < output.cols; j++) {	
				float dist = 1000000;
				for (int k = 0; k < K; k++) {
					float newdist = squaredeEuclidDist(Point3i(j,i,-1), centers.at(k));
					if (dist > newdist) {
						dist = newdist;
						voronoi.at<Vec3b>(i, j) = colors.at(centers.at(k).z);
					}
				}
			}
		}

		std::cout << "Nr of iterations is " << executionCycles;

		imshow("orig img", src);
		imshow("kmeans img", output);
		imshow("voronoi", voronoi);

		// Wait until user press some key
		waitKey(0);
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Open a file for reading \n");
		printf(" 11 - Display points from a file \n");
		printf(" 12 - Gradient descent \n");
		printf(" 13 - L2 Ransac\n");
		printf(" 14 - L3 Hough\n");
		printf(" 15 - L4 Chamfer\n");
		printf(" 16 - L4 ChamferDifference\n");
		printf(" 17 - L5 Statistical data analysis\n");
		printf(" 18 - L6 K Means\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);

		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testParcurgereSimplaDiblookStyle(); //diblook style
			break;
		case 4:
			//testColor2Gray();
			testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;
		case 10:
			openAndReadFile();
			break;
		case 11:
			displayPoints();
			break;
		case 12:
			gradientDescent();
			break;
		case 13:
			ransac();
			break;
		case 14:
			haugh();
			break;
		case 15:
			ChamferDistance();
			break;
		case 16:
			CalculateChamDifference();
			break;
		case 17:
			statisticalAnalisys();
			break;
		case 18:
			KMeans();
			break;
		}


	} while (op != 0);
	return 0;
}