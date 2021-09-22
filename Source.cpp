#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/core/core.hpp>

#include <iostream>
#include <string>
#include <vector>



int main()
{
	std::string pathImgSkins = "Resources/skins.png";
	std::string pathImgTest2 = "Resources/1000s_d_850.jpg";

	cv::Mat imgSkins, imgTest1, imgTest2;
	cv::Mat imgSkinsHSV, imgTest1HSV, imgTest2HSV;

	imgSkins = cv::imread(pathImgSkins, cv::IMREAD_COLOR);
	cv::resize(imgSkins, imgSkins, cv::Size(600, 400));

	if (imgSkins.empty())
	{
		std::cerr << "Error. Image \"skins.png\" hasn`t downloaded\n";

		return EXIT_FAILURE;
	}

	imgTest1 = cv::imread(pathImgTest2, cv::IMREAD_COLOR);
	cv::resize(imgTest1, imgTest1, cv::Size(700, 600));

	if (imgTest1.empty())
	{
		std::cerr << "Error. Image \"skins.png\" hasn`t downloaded\n";

		return EXIT_FAILURE;
	}

	cv::cvtColor(imgSkins, imgSkinsHSV, cv::COLOR_BGR2HSV);
	cv::cvtColor(imgTest1, imgTest1HSV, cv::COLOR_BGR2HSV);


	// Обчислення гістограм 
	//----------------------------------//

	cv::Mat histogram;

	// Підказка https://docs.opencv.org/3.4/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d

	int hueBins = 180, saturBins = 256;
	int histSize[] = { hueBins, saturBins };
	int channelNumbers[] = { 0 , 1 };
	float range1[] = { 0.0f, 180.0f };
	float range2[] = { 0.0f, 256.0f };
	const float *histRanges[] = { range1 , range2 };

	bool uniform = true;
	bool accumulate = false;

	cv::calcHist(&imgSkinsHSV, 1, channelNumbers, cv::Mat(), histogram, 2, histSize, histRanges, uniform, accumulate);

	double maxVal = 0.0;
	//Знаходимо глобальний максимум
	cv::minMaxLoc(histogram, 0, &maxVal, 0, 0);

	std::cout << "Global Max = " << maxVal << '\n';

	//----------------------------------//


	// Рисуємо гістаграму
	//----------------------------------//

	int scale = 4;
	cv::Mat histImage = cv::Mat::zeros(saturBins*scale, hueBins*scale, CV_8UC3);

	for (int h = 0; h < hueBins; h++)
	{
		for (int s = 0; s < saturBins; s++)
		{
			//знаходжу значення на гістаграмі у точці (h, s)
			float binVal = histogram.at<float>(h, s);
			//if(s >= 6 && s <= 15)
			//std::cout << "binVal[" << h << "][" << s << "] = " << binVal << '\n';
			// Обраховую інтенсивність кольору
			int intensity = cvRound(binVal * 255 / maxVal);

			cv::rectangle(histImage, cv::Point(h*scale, s*scale),
				cv::Point((h + 1)*scale - 1, (s + 1)*scale - 1), cv::Scalar::all(intensity), -1);
		}
	}

	//----------------------------------//


	// Знаходження маски
	//----------------------------------//

	cv::Mat backproj;
	cv::calcBackProject(&imgTest1HSV, 1, channelNumbers, histogram, backproj, histRanges, 1, true);

	cv::Mat backprojFiltered, imgDil;
	cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	cv::erode(backproj, imgDil, kernel1);
	cv::filter2D(imgDil, backprojFiltered, -1, kernel2);
	cv::threshold(backprojFiltered, backprojFiltered, 100, 255, cv::THRESH_BINARY);

	//----------------------------------//



	cv::imshow("Img skins", imgSkins);
	cv::imshow("Histigram", histImage);
	cv::imshow("Image", imgTest1);
	cv::imshow("Mask", backproj);
	cv::imshow("Mask Filtered", backprojFiltered);


	cv::waitKey(0);
	return 0;
}