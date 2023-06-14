// PanelFunDectDLL.cpp : ���� DLL Ӧ�ó���ĵ���������
//

#include "stdafx.h"
#include "PanelFunDectDLL.h"

// ���ǵ���������һ��ʾ��
PANELFUNDECTDLL_API int nPanelFunDectDLL=0;

// ���ǵ���������һ��ʾ����
 PANELFUNDECTDLL_API bool fnPanelFunDectDLL(void)
{
    return abs(1-9)>=100;
}

// �����ѵ�����Ĺ��캯����
// �й��ඨ�����Ϣ������� PanelFunDectDLL.h
CPanelFunDectDLL::CPanelFunDectDLL()
{
    return;
}
/**
 �������أ������ȷ������dev����ֵ
* */
int getEveryPixels(Mat& inputImage) {

	int rowNumbers = inputImage.rows;
	int colNumbers = inputImage.cols;
	long avg = rowNumbers * colNumbers;
	int sum = 0;
	for (int i = 0; i < rowNumbers; i++)
	{
		for (int j = 0; j < colNumbers; j++)
		{
			// v Value ����
			sum += inputImage.at<Vec3b>(i, j)[1];
		}
	}
	return sum / avg;

}

//�ĸ�������Ͻǣ����Ͻǣ����£����£�w��h��У�����ͼ���͸ߣ�
/*
����������У��ͼ��
srcPath : ԭͼ��·��
Point���� �� ԭͼ��У��������ĸ������ꣻ�����Ͻǣ����Ͻǣ����£�����
width��height �� У����ͼ��Ŀ�͸ߣ�
*/
extern "C" PANELFUNDECTDLL_API int transformImg(const char * srcPath, const char * dstPath, float firstPointX, float firstPointY, float secondPointX, float secondPointY, float thridPointX, float thridPointY, float fourPointX, float fourPointY, float width, float height)
{
	//У��ͼ��
	Mat img = imread(srcPath);
	Mat matrix, imgWarp;
	Point2f src[4] = { { firstPointX,firstPointY },{ secondPointX,secondPointY },{ thridPointX,thridPointY },{ fourPointX,fourPointY } };
	Point2f dst[4] = { { 0.0f,0.0f },{ width,0.0f },{ 0.0f,height },{ width,height } };
	matrix = getPerspectiveTransform(src, dst);
	warpPerspective(img, imgWarp, matrix, Point(width,height));
	imwrite(dstPath, imgWarp);
	return 1;
}





/*
������������Ļ�������Ӽ��
srcPath : У�����ԭͼ
comparePath �� У����ĵ�ǰͼ
x,y��w,h �� �ü��������꣬���Ͻǵ����꣬��ȣ����ȣ�
threshold : ���Ȳ�ֵ����ֵ
*/
extern "C" PANELFUNDECTDLL_API int brightPulsDetection(const char * srcPath,const char * comparePath,int x,int y,int w,int h,int threshold)
{
	 /*��ȡ��Ƭ*/
	
	Mat srcImg = imread(srcPath);
	Mat compareImg = imread(comparePath);
	Rect roi(x, y, w, h);
	Mat srcImgCrop = srcImg(roi);
	Mat compareImgCrop = compareImg(roi);
	 //Ԥ��������ɫ�ʿռ�ı���
	Mat srcImg_out_withHSI;
	Mat compareImg_out_withHSI;
	// ת��ɫ�ʿռ䵽HSVɫ�ʿռ�
	cvtColor(srcImgCrop, srcImg_out_withHSI, CV_BGR2HLS);
	cvtColor(compareImgCrop, compareImg_out_withHSI, CV_BGR2HLS);
	// ��HSIɫ�ʿռ������
	int srcAvg = getEveryPixels(srcImg_out_withHSI);
	int compareAvg = getEveryPixels(compareImg_out_withHSI);   //����ʵ�ʲɼ�ͼ����Ա�ģ��ͼ�����ȵĲ�ֵ��ֵ�����ж�
	int difference = compareAvg - srcAvg;
	if (difference >= threshold) {
		return 1;
	}
	return 0;
	
}


//��Ļ����-
/*
������������Ļ���ȼ��ټ��
srcPath : У����Ĳο�ͼ
comparePath �� У����ĵ�ǰͼ
x,y��w,h �� �ü��������꣬���Ͻǵ����꣬��ȣ����ȣ�
threshold : ���Ȳ�ֵ����ֵ
*/
extern "C" PANELFUNDECTDLL_API int brightReduceDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int threshold)
{
	/*��ȡ��Ƭ*/

	Mat srcImg = imread(srcPath, 1);
	Mat compareImg = imread(comparePath, 1);
	Rect roi(x, y, w, h);
	Mat srcImgCrop = srcImg(roi);
	Mat compareImgCrop = compareImg(roi);
	//Ԥ��������ɫ�ʿռ�ı���
	Mat srcImg_out_withHSI;
	Mat compareImg_out_withHSI;
	// ת��ɫ�ʿռ䵽HSVɫ�ʿռ�
	cvtColor(srcImgCrop, srcImg_out_withHSI, COLOR_BGR2HLS);
	cvtColor(compareImgCrop, compareImg_out_withHSI, COLOR_BGR2HLS);
	// ��HSIɫ�ʿռ������
	int srcAvg = getEveryPixels(srcImg_out_withHSI);
	int compareAvg = getEveryPixels(compareImg_out_withHSI);
	int difference = srcAvg - compareAvg;
	if (difference >= threshold) {
		return 1;
	}
	return 0;
}



//��ɫ��⣬���̱�죻���ǲü��������겻ͬ
/*
������������ɫ���
prePath : У����Ĳο�ͼ
srcPath ��У����ĵ�ǰͼ
x,y��w,h���ü��������꣬���Ͻǵ����꣬��ȣ����ȣ�
greenThreshold : Gͨ������ֵ��ǰΪ200��redThreshold:Rͨ������ֵֵ������Ϊ��Ӧ��ʵ���������
cntThreshold : ����ͨ����ֵ�ĵ����,Ĭ��Ϊ3000������ֵ0,7000��
*/
extern "C" PANELFUNDECTDLL_API int  colorDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold,int cntThreshold)
{
	Mat srcImg = imread(srcPath);
	Mat compareImg = imread(comparePath);
	Rect roi(x, y, w, h);              //Rect:����һ�����δ���roiΪ����Ȥ������            
	Mat srcImgCrop = srcImg(roi);
	Mat compareImgCrop = compareImg(roi);
	int rowNumber = srcImgCrop.rows;
	int colNumber = srcImgCrop.cols;
	int srcCount = 0;
	int compareCount = 0;
	for (int i = 0; i < rowNumber; i++)
	{
		for (int j = 0; j < colNumber; j++)
		{
			//Vec3b������Ϊ3��vector��������������Ϊunchar 
			if (srcImgCrop.at<Vec3b>(i, j)[2] > redThreshold) {            //ʵ��ͼ��ͨ����ֵ�Ķ�Ӧ��������ص������
				srcCount++;
			}
			if (compareImgCrop.at<Vec3b>(i, j)[2] > redThreshold) {        //�Ա�ͼ��
				compareCount++;
			}
		}
	}
	if (abs(compareCount - srcCount) >= cntThreshold) {
		return 1;
	}
	return 0;
}

//HOME,��һ������Դ������һ����back,����+������-�����ǲü��������겻ͬ

/*
	����������HOME���
	prePath : У����Ĳο�ͼ
	srcPath ��У����ĵ�ǰͼ
	x,y,w,h ���ü��������꣬���Ͻǵ����꣬��ȣ����ȣ�
	greenThreshold : Gͨ������ֵ��ǰΪ100��
	cntThreshold : ����ͨ����ֵ�ĵ����,Ĭ��Ϊ3000������ֵ0,7000��
*/
extern "C" PANELFUNDECTDLL_API int homeDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold)
{
	return colorDetection(srcPath, comparePath, x,  y,  w,  h,  redThreshold, cntThreshold);
	
}
/*
������������һ�����
prePath : У����Ĳο�ͼ
srcPath ��У����ĵ�ǰͼ
x,y,w,h ���ü��������꣬���Ͻǵ����꣬��ȣ����ȣ�
greenThreshold : Gͨ������ֵ��ǰΪ100��
cntThreshold : ����ͨ����ֵ�ĵ����,Ĭ��Ϊ3000������ֵ0,7000��
*/
extern "C" PANELFUNDECTDLL_API int songUpDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold)
{
	return colorDetection(srcPath, comparePath, x, y, w, h, redThreshold, cntThreshold);

}
/*
������������һ�����
prePath : У����Ĳο�ͼ
srcPath ��У����ĵ�ǰͼ
x,y,w,h ���ü��������꣬���Ͻǵ����꣬��ȣ����ȣ�
greenThreshold : Gͨ������ֵ��ǰΪ100��
cntThreshold : ����ͨ����ֵ�ĵ����,Ĭ��Ϊ3000������ֵ0,7000��
*/
extern "C" PANELFUNDECTDLL_API int songDownDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold)
{
	return colorDetection(srcPath, comparePath, x, y, w, h, redThreshold, cntThreshold);

}
/*
�����������������Ӽ��
prePath : У����Ĳο�ͼ
srcPath ��У����ĵ�ǰͼ
x,y,w,h ���ü��������꣬���Ͻǵ����꣬��ȣ����ȣ�
greenThreshold : Gͨ������ֵ��ǰΪ100��
cntThreshold : ����ͨ����ֵ�ĵ����,Ĭ��Ϊ3000������ֵ0,7000��
*/
extern "C" PANELFUNDECTDLL_API int volumeUpDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold)
{
	return colorDetection(srcPath, comparePath, x, y, w, h, redThreshold, cntThreshold);

}
/*
�����������������ټ��
prePath : У����Ĳο�ͼ
srcPath ��У����ĵ�ǰͼ
x,y,w,h ���ü��������꣬���Ͻǵ����꣬��ȣ����ȣ�
greenThreshold : Gͨ������ֵ��ǰΪ100��
cntThreshold : ����ͨ����ֵ�ĵ����,Ĭ��Ϊ3000������ֵ0,7000��
*/
extern "C" PANELFUNDECTDLL_API int volumeDownDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold)
{
	return colorDetection(srcPath, comparePath, x, y, w, h, redThreshold, cntThreshold);

}


/*
������������Դ�����
prePath : У����Ĳο�ͼ
srcPath �� У����ĵ�ǰͼ
x,y,w,h �� �ü��������꣬���Ͻǵ����꣬��ȣ����ȣ�
greenThreshold : Gͨ������ֵ��ǰΪ100��
cntThreshold : ����ͨ����ֵ�ĵ����,Ĭ��Ϊ3000������ֵ0,7000��
*/
extern "C" PANELFUNDECTDLL_API int powerDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold)
{
	return colorDetection(srcPath, comparePath, x, y, w, h, redThreshold, cntThreshold);
}
/*
����������back�����
prePath : У����Ĳο�ͼ
srcPath �� У����ĵ�ǰͼ
x,y,w,h �� �ü��������꣬���Ͻǵ����꣬��ȣ����ȣ�
greenThreshold : Gͨ������ֵ��ǰΪ100��
cntThreshold : ����ͨ����ֵ�ĵ����,Ĭ��Ϊ3000������ֵ0,7000��
*/
extern "C" PANELFUNDECTDLL_API int backDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold)
{
	return colorDetection(srcPath, comparePath, x, y, w, h, redThreshold, cntThreshold);
}
/*
�����������˳������
srcPath �� У����ĵ�ǰͼ
x,y,w,h �� �ü��������꣬���Ͻǵ����꣬��ȣ����ȣ�
greenThreshold : Gͨ������ֵ��
cntThreshold : ����ͨ����ֵ�ĵ����,
*/
extern "C" PANELFUNDECTDLL_API int exitDetection(const char * srcPath, int x, int y, int w, int h, int redThreshold, int cntThreshold)
{
	Mat srcImg = imread(srcPath);
	Rect roi(x, y, w, h);
	Mat srcImgCrop = srcImg(roi);
	int rowNumber = srcImgCrop.rows;
	int colNumber = srcImgCrop.cols;
	int srcCount = 0;
	for (int i = 0; i < rowNumber; i++)
	{
		for (int j = 0; j < colNumber; j++)
		{
			if (srcImgCrop.at<Vec3b>(i, j)[2] > redThreshold) {
				srcCount++;
			}
		}
	}
	if (srcCount >= cntThreshold) {
		return 1;
	}
	return 0;
}

/*
�����������򿪲���������
srcPath : У����ĵ�ǰͼ
x,y,w,h �� �ü��������꣬���Ͻǵ����꣬��ȣ����ȣ�
greenThreshold : Gͨ����
��ֵ��
cntThreshold : С��ͨ����ֵ�ĵ����
*/
extern "C" PANELFUNDECTDLL_API int openAppDetection(const char * srcPath , int x, int y, int w, int h, int redThreshold, int cntThreshold)
{
	Mat srcImg = imread(srcPath);
	Rect roi(x, y, w, h);
	Mat srcImgCrop = srcImg(roi);
	int rowNumber = srcImgCrop.rows;
	int colNumber = srcImgCrop.cols;
	int srcCount = 0;
	for (int i = 0; i < rowNumber; i++)
	{
		for (int j = 0; j < colNumber; j++)
		{
			if (srcImgCrop.at<Vec3b>(i, j)[2] < redThreshold) {
				srcCount++;
			}
		}
	}
	if (srcCount >= cntThreshold) {
		return 1;
	}
	return 0;
}
/*
����������Bͨ����ɫ���
prePath : У����Ĳο�ͼ
srcPath �� У����ĵ�ǰͼ
x,y��w,h �� �ü��������꣬���Ͻǵ����꣬��ȣ����ȣ�
greenThreshold : Gͨ������ֵ��ǰΪ50��
cntThreshold : ����ͨ����ֵ�ĵ����,Ĭ��Ϊ3000������ֵ0,7778��
*/
extern "C" PANELFUNDECTDLL_API int greenColorDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int greenThreshold, int cntThreshold)
{
	Mat srcImg = imread(srcPath);
	Mat compareImg = imread(comparePath);
	Rect roi(x, y, w, h);
	Mat srcImgCrop = srcImg(roi);
	Mat compareImgCrop = compareImg(roi);
	int rowNumber = srcImgCrop.rows;
	int colNumber = srcImgCrop.cols;
	int srcCount = 0;
	int compareCount = 0;
	for (int i = 0; i < rowNumber; i++)
	{
		for (int j = 0; j < colNumber; j++)
		{

			if (srcImgCrop.at<Vec3b>(i, j)[1] > greenThreshold) {
				srcCount++;
			}
			if (compareImgCrop.at<Vec3b>(i, j)[1] > greenThreshold) {
				compareCount++;
			}
		}
	}
	if (abs(compareCount - srcCount) >= cntThreshold) {
		return 1;
	}
	return 0;
}

/*
������������������
prePath : У����Ĳο�ͼ
srcPath �� У����ĵ�ǰͼ
x,y,w,h �� �ü��������꣬���Ͻǵ����꣬��ȣ����ȣ�
greenThreshold : Gͨ������ֵ��ǰΪ100��
cntThreshold : ����ͨ����ֵ�ĵ����,Ĭ��Ϊ3000������ֵ0,7000��
*/
extern "C" PANELFUNDECTDLL_API int  backLightDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int greenThreshold, int cntThreshold)
{
	return greenColorDetection(srcPath, comparePath, x, y, w, h, greenThreshold, cntThreshold);
}


/*
��������������ʶ����
prePath : У����Ĳο�ͼ
srcPath �� У����ĵ�ǰͼ
x,y,w,h �� �ü��������꣬���Ͻǵ����꣬��ȣ����ȣ�
binThreshold : ��ֵ������ֵ��ǰΪ25��
cntThreshold : ����ͨ����ֵ�ĵ����,Ĭ��Ϊ3000������ֵ0,7000��
*/
extern "C" PANELFUNDECTDLL_API int numRecognition(const char * srcPath, const char * numModelPath,int x, int y, int w, int h, int binThreshold)
{
	Mat srcImg = imread(srcPath);
	Rect roi(x, y, w, h);
	Mat srcImgCrop = srcImg(roi);
	if (!srcImgCrop.data) {
		return -1;
	}
	//��ͼ����д���ת��Ϊ�Ҷ�ͼȻ����תΪ��ֵͼ
	Mat grayImage;
	cvtColor(srcImgCrop, grayImage, CV_BGR2GRAY);
	Mat binImage;
	//��4������ΪCV_THRESH_BINARY_INV����Ϊ�ҵ�����ԭͼΪ�׵׺���
	//��Ϊ�ڵװ�����ѡ��CV_THRESH_BINARY����
	//0-5��ֵΪ5,6��7����ֵΪ25
	//��һ��������img����Դͼ��
	/* ������������255����������ֵ�Ĳ���ȡֵ�Ƕ��٣�����cv.THRESH_BINARY���ԣ�
	���ĸ�������cv.ADAPTIVE_THRESH_MEAN_C����
	��1����һ�������ڼ�����ֵ�����õ��㷨��������ȡֵ���ֱ�Ϊ ADAPTIVE_THRESH_MEAN_C �� ADAPTIVE_THRESH_GAUSSIAN_C
	��2��ADAPTIVE_THRESH_MEAN_C�ļ��㷽���Ǽ���������ƽ��ֵ�ټ�ȥ���߸�����2��ֵ��
	��3��ADAPTIVE_THRESH_GAUSSIAN_C�ļ��㷽���Ǽ��������ĸ�˹��ֵ�ټ�ȥ���߸�����2��ֵ
	�����������cv.THRESH_BINARY����������ֵ���ͣ�ֻ������ȡֵ���ֱ�Ϊ THRESH_BINARY ��THRESH_BINARY_INV
	������������11����adaptiveThreshold�ļ��㵥λ�����ص�������Сѡ�����Ǿֲ������С��3��5��7��
	���߸�������2�����������ʵ������һ��ƫ��ֵ���������þ�ֵ�͸�˹������ֵ���ټ�������ֵ����������ֵ��*/
	adaptiveThreshold(~grayImage, binImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 21, 4);   //����Ӧ��ֵ������Ӧ��ֵ��ÿһ�����ص����ֵ�ǲ�ͬ�ģ�����ֵ����������ͼ�����ش����Ȩƽ��������
	Mat binImage1 = binImage.clone();
	
	string numPath(numModelPath);
	numPath = numPath + "/%d.jpg";
	const char* model_filename = NULL;
	model_filename = numPath.c_str();
	//����ģ��
	vector<Mat> myTemplate;
	for (int i = 0; i < 8; i++)
	{
		char name[128];
		sprintf_s(name, model_filename, i);
		Mat temp = imread(name, 0);
		myTemplate.push_back(temp);
	}
	// ��˳��ȡ���ͷָ�����
	

	//���бȽ�,��ͼƬ��ģ�������Ȼ����ȫ�����غͣ�����С��ʾԽ���ƣ��������ƥ��
	//MatchTemplate����������һ��ͼ����Ѱ������һ��ģ��ͼ����ƥ��(����)���֡�
	/*image������һ����ƥ���ͼ��֧��8U����32F��
	  templ������һ��ģ��ͼ����image��ͬ���͡�
	  result������������ľ���32F���͡�
	  method��Ҫʹ�õ����ݱȽϷ�����*/
	vector<int> seq;  //˳����ʶ����
	Mat res;
	double max = 0;
	int max_seq = 0;
	vector<double> ans;
	double a = 0;
	double b = 0;
	for (int j = 5; j < 8; j++)
	{
		matchTemplate(myTemplate[j], binImage, res, TM_CCOEFF_NORMED);  //һ��ͼ������һ��ͼ���ƥ��ĵط�
		//myTemplat������һ����ƥ���ͼ��֧��8U����32F��8Uָ����8λ�޷�������32ֵ����32λ�ĸ�����
	    //binImage������һ��ģ��ͼ����image��ͬ���͡�
	    //res������������ľ���32F���͡�
	    //TM_CCOEFF_NORMED��Ҫʹ�õ����ݱȽϷ�����
		
		minMaxLoc(res, &a, &b);  //Ѱ��ͼ������С���ֵ
		ans.push_back(b);
		if (b > max)            //�Ƚ�����ƥ���������ƶ����ֵ
		{
			max = b;
			max_seq = j;
		}
	}
	if (max < 0.3)
	{
		max_seq = 1;
	}
	seq.push_back(max_seq);
	return seq[0];
}


//������ǿ
Mat bright(Mat image, float alpha, int beta)
{
	for (int i = 0; i < image.rows; i++)
		for (int j = 0; j < image.cols; j++) {
			for (int k = 0; k < 3; k++) {
				int tmp = (uchar)image.at<Vec3b>(i, j)[k] * alpha + beta;
				if (tmp > 255)
					image.at<Vec3b>(i, j)[k] = 2 * 255 - tmp;
				else
					image.at<Vec3b>(i, j)[k] = tmp;
			}
		}
	return image;
}

//�и�ͼ��imageΪ���и�ͼ��xywdΪ�и�����
Mat cutImage(Mat image, int x, int y, int w, int h)
{
	Mat img = image.clone();
	Rect rect(x, y, w, h);
	Mat ROI = img(rect);
	return ROI;
}
//����ƽ�����ȣ�inputImageΪ�����ͼ��inputImage2Ϊģ�壬tΪ����ɸѡ��ֵ��
int getEveryPixels(Mat& inputImage, Mat& inputImage2, int t)
{
	int rowNumbers = inputImage.rows;
	int colNumbers = inputImage.cols;
	long avg = rowNumbers * colNumbers;
	int sum = 0;
	int num = 0;
	for (int i = 0; i < rowNumbers; i++)
	{
		for (int j = 0; j < colNumbers; j++)
		{
			if (inputImage2.at<Vec3b>(i, j)[1] > t) {
				sum += inputImage.at<Vec3b>(i, j)[1];
				num++;

			}


		}
	}

	if (num == 0)
	{
		return 0;
	}
	else
	{
		return sum / num;      //����һ��ƽ������ֵ
	}
}

/*
���˿ӡ���ȱ仯
srcPath���仯ǰͼƬ
srcPath���仯��ͼƬ
x,y,w,h���ü��������꣬���Ͻǵ����꣬��ȣ�����
t0������ɸѡ��ֵ��Ĭ��Ϊ30
t1�����ȱ仯��ֵ
ע��t1Ĭ��Ϊ����
*/
extern "C" PANELFUNDECTDLL_API int liangdubianhua(const char* srcPath, const char* srcPath2, const char* srcPath3, int x, int y, int w, int h, int t0, int t1)
{
	//��ȡͼ��
	Mat img_before = imread(srcPath, 1);
	Mat img_after = imread(srcPath2, 1);
	Mat img_model = imread(srcPath2, 1);//��Ϊģ��ͼ��ɸѡ����ʱʹ�ã���洢·���̶�
	
	//�и�ͼ��
	Mat img_cut_before = cutImage(img_before, x, y, w, h);
	Mat img_cut_after = cutImage(img_after, x, y, w, h);
	Mat img_cut_model = cutImage(img_model, x, y, w, h);

	bright(img_cut_before, 15.0, 5);
	bright(img_cut_after, 15.0, 5);
	bright(img_cut_model, 15.0, 5);

	//RGBתΪHSI
	Mat img_before_HSI;
	Mat img_after_HSI;
	Mat img_model_HSI;
	cvtColor(img_cut_before, img_before_HSI, COLOR_BGR2HLS);
	cvtColor(img_cut_after, img_after_HSI, COLOR_BGR2HLS);
	cvtColor(img_cut_model, img_model_HSI, COLOR_BGR2HLS);

	//����ƽ������
	int avg_before = getEveryPixels(img_before_HSI, img_model_HSI, t0);
	int avg_after = getEveryPixels(img_after_HSI, img_model_HSI, t0);

	//�Ƚ����Ȳ��������1Ϊ�ϸ�0Ϊ���ϸ�
	int difference = abs(avg_after - avg_before);   //������ǰ����ģ����жԱȵõ���ֵ
	if (difference >= t1 )
	{
		return 1;
	}
	else
	{
		return 0;
	}
}


//����ƽ�����ȣ�ר����һ���Լ�⣬imageΪ����ͼƬ��tΪ����ɸѡ��ֵ
//ע��
//����һ���Լ������ģ�����ɸѡ���ص㣬��˱�����ר����һ���Լ��
int getliangdu(Mat& image, int t)
{
	int rowNumbers = image.rows;
	int colNumbers = image.cols;
	long avg = rowNumbers * colNumbers;
	int sum = 0;
	int num = 0;
	for (int i = 0; i < rowNumbers; i++)
	{
		for (int j = 0; j < colNumbers; j++)
		{
			if (image.at<Vec3b>(i, j)[1] > t) {
				sum += image.at<Vec3b>(i, j)[1];
				num++;
			}
		}
	}
	if (num == 0)
	{
		return 0;
	}
	else
	{
		return sum / num;
	}
	
}

//����ͼƬ��ָ����Ϣ
string ImageHashValue(IplImage* src)
{
	string resStr(256, '\0');
	IplImage* image = cvCreateImage(cvGetSize(src), src->depth, 1);
	//step one : �ҶȻ�
	if (src->nChannels == 3) cvCvtColor(src, image, CV_BGR2GRAY);
	else cvCopy(src, image);
	//step two : ��С�ߴ� 16*16
	IplImage* temp = cvCreateImage(cvSize(16, 16), image->depth, 1);
	cvResize(image, temp);
	//step three : ��ɫ��
	uchar* pData;
	for (int i = 0; i < temp->height; i++)
	{
		pData = (uchar*)(temp->imageData + i * temp->widthStep);
		for (int j = 0; j < temp->width; j++)
			pData[j] = pData[j] / 4;
	}
	//step four : ����ƽ���Ҷ�ֵ
	double average = cvAvg(temp).val[0];
	//step five : �����ϣֵ
	int index = 0;
	for (int i = 0; i < temp->height; i++)
	{
		pData = (uchar*)(temp->imageData + i * temp->widthStep);
		for (int j = 0; j < temp->width; j++)
		{
			if (pData[j] >= average)
				resStr[index++] = '1';
			else
				resStr[index++] = '0';
		}
	}
	return resStr;
}

//����ָ����Ϣ��������ͼ������ƶ�
double ImageSimilarity(string& str1, string& str2)
{
	double similarity = 1.0;
	for (int i = 0; i < 256; i++)
	{
		char c1 = str1[i];
		char c2 = str2[i];
		if (c1 != c2)
		similarity = similarity - 1.0 / 256;
	}
	return similarity;
}

//���ƶȱȽ�
double ImageCompared(Mat input1, Mat input2) {
	IplImage imgTmp1 = cvIplImage(input1);
	IplImage* image1 = cvCloneImage(&imgTmp1);      //
	IplImage imgTmp2 = cvIplImage(input2);
	IplImage* image2 = cvCloneImage(&imgTmp2);
	//cvShowImage("image1", image1);
	//cvShowImage("image2", image2);


	string imgPrint1 = ImageHashValue(image1);
	string imgPrint2 = ImageHashValue(image2);
	double similarity = ImageSimilarity(imgPrint1, imgPrint2);
	return similarity;
}



//�������׼λ�õ�ƫ����룬��λ������
int drawSquare(Mat Image, int x0/*��׼λ��x����*/, int y0/*��׼λ��y����*/)
{
	Mat srcImg = Image.clone();
	Mat dstImg = srcImg.clone();
	Mat medianImg;
	medianBlur(srcImg, medianImg, 5);
	//imshow("��ֵ�˲�", medianImg);
	Mat brightImg = medianImg.clone();
	//bright(brightImg, 5.0, 5);
	//imshow("��������", brightImg);

	cvtColor(brightImg, brightImg, CV_BGR2GRAY);
	Mat binaryImg;
	adaptiveThreshold(~brightImg, binaryImg, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 17, 6);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarcy;
	findContours(binaryImg, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<Rect> boundRect(contours.size());  //������Ӿ��μ���

	vector<RotatedRect> box(contours.size()); //������С��Ӿ��μ���
	Point2f rect[4];
	for (int i = 0; i < contours.size(); i++)
	{
		box[i] = minAreaRect(Mat(contours[i]));  //����ÿ��������С��Ӿ���
		boundRect[i] = boundingRect(Mat(contours[i]));
		//circle(dstImg, Point(box[i].center.x, box[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  //������С��Ӿ��ε����ĵ�
		box[i].points(rect);  //����С��Ӿ����ĸ��˵㸴�Ƹ�rect����
		//rectangle(dstImg, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(255, 0, 0), 2, 8);
		for (int j = 0; j < 4; j++)
		{
			//line(dstImg, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //������С��Ӿ���ÿ����
		}

	}
	for (int i = 0; i < boundRect.size(); i++) {
		if (boundRect[i].width < 10 && boundRect[i].height < 10) {
			for (int m = boundRect[i].x; m < boundRect[i].x + boundRect[i].width; m++)
				for (int n = boundRect[i].y; n < boundRect[i].y + boundRect[i].height; n++) {
					binaryImg.at<uchar>(n, m) = 0;

				}
		}
	}





	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	morphologyEx(binaryImg, binaryImg, MORPH_CLOSE, element);


	findContours(binaryImg, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);


	vector<RotatedRect> box1(contours.size()); //������С��Ӿ��μ���

	for (int i = 0; i < contours.size(); i++)
	{
		box1[i] = minAreaRect(Mat(contours[i]));  //����ÿ��������С��Ӿ���
		boundRect[i] = boundingRect(Mat(contours[i]));
		circle(dstImg, Point(box1[i].center.x, box1[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  //������С��Ӿ��ε����ĵ�
		box1[i].points(rect);  //����С��Ӿ����ĸ��˵㸴�Ƹ�rect����
		rectangle(dstImg, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(255, 0, 0), 2, 8);
		for (int j = 0; j < 4; j++)
		{
			line(dstImg, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //������С��Ӿ���ÿ����
		}
		circle(dstImg, Point(rect[2]), 5, Scalar(255, 0, 0), -1, 8);
	}
	if (boundRect.size() == 0) {
		int d = pow(pow((0 - x0), 2) + pow((0 - y0), 2), 0.5);
		return d;
	}
	int distance = 0;
	distance = pow(pow((boundRect[0].x - x0), 2) + pow((boundRect[0].y - y0), 2), 0.5);

	return distance;


}

Mat getsquare(Mat Image)
{
	Mat srcImg = Image.clone();
	Mat dstImg = srcImg.clone();
	Mat medianImg;
	medianBlur(srcImg, medianImg, 5);

	Mat brightImg = medianImg.clone();


	cvtColor(brightImg, brightImg, CV_BGR2GRAY);
	Mat binaryImg;
	adaptiveThreshold(~brightImg, binaryImg, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 17, 6);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarcy;
	findContours(binaryImg, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<Rect> boundRect(contours.size());  //������Ӿ��μ���

	vector<RotatedRect> box(contours.size()); //������С��Ӿ��μ���
	Point2f rect[4];
	for (int i = 0; i < contours.size(); i++)
	{
		box[i] = minAreaRect(Mat(contours[i]));  //����ÿ��������С��Ӿ���
		boundRect[i] = boundingRect(Mat(contours[i]));
		//circle(dstImg, Point(box[i].center.x, box[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  //������С��Ӿ��ε����ĵ�
		box[i].points(rect);  //����С��Ӿ����ĸ��˵㸴�Ƹ�rect����
		//rectangle(dstImg, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(255, 0, 0), 2, 8);
		for (int j = 0; j < 4; j++)
		{
			//line(dstImg, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //������С��Ӿ���ÿ����
		}

	}
	for (int i = 0; i < boundRect.size(); i++) {
		if (boundRect[i].width < 10 && boundRect[i].height < 10) {
			for (int m = boundRect[i].x; m < boundRect[i].x + boundRect[i].width; m++)
				for (int n = boundRect[i].y; n < boundRect[i].y + boundRect[i].height; n++) {
					binaryImg.at<uchar>(n, m) = 0;

				}
		}
	}




	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	morphologyEx(binaryImg, binaryImg, MORPH_CLOSE, element);


	findContours(binaryImg, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);


	vector<RotatedRect> box1(contours.size()); //������С��Ӿ��μ���

	for (int i = 0; i < contours.size(); i++)
	{
		box1[i] = minAreaRect(Mat(contours[i]));  //����ÿ��������С��Ӿ���
		boundRect[i] = boundingRect(Mat(contours[i]));
		circle(dstImg, Point(box1[i].center.x, box1[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  //������С��Ӿ��ε����ĵ�
		box1[i].points(rect);  //����С��Ӿ����ĸ��˵㸴�Ƹ�rect����
		rectangle(dstImg, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(255, 0, 0), 2, 8);
		for (int j = 0; j < 4; j++)
		{
			line(dstImg, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //������С��Ӿ���ÿ����
		}
		circle(dstImg, Point(rect[2]), 5, Scalar(255, 0, 0), -1, 8);
	}


	if (boundRect.size() == 0) {
		Mat img = cutImage(Image, 0, 0, 0, 0);
		return img;
	}

	int x = boundRect[0].x;
	int y = boundRect[0].y;
	int w = boundRect[0].width;
	int h = boundRect[0].height;
	Mat img = cutImage(Image, x, y, w, h);
	return img;
}

int getRes(Mat mat,Mat HSI)
{
	int rowNumbers = mat.rows;
	int colNumbers = mat.cols;
	int sum = 0, cnt = 1;
	vector<Point> contours;
	for (int i = 0; i < rowNumbers; i++)
	{
		for (int j = 0; j < colNumbers; j++)
		{
			if (mat.at<uchar>(i, j) == 255)
			{
				cnt++;
				sum += HSI.at<Vec3b>(i, j)[1];
			}
		}
	}
	return sum / cnt;
}

/*����һ���Լ��
srcPath�������ͼ��
srcPath2��ģ��ͼ�����ڼ�������ʱɸѡ���ص�
x1,y1,w1,h1:home���������꣬���Ͻ����꣬��ȣ�����
x1,y1,w1,h1:��һ�����������꣬���Ͻ����꣬��ȣ�����
x1,y1,w1,h1:��һ�����������꣬���Ͻ����꣬��ȣ�����
x1,y1,w1,h1:back���������꣬���Ͻ����꣬��ȣ�����
t0:����ɸѡ��ֵ
t1:ƫ��ٷֱ���ֵ
*/
extern "C" PANELFUNDECTDLL_API int yizhixing(const char* srcPath, const char* srcPath_model, int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2, int x3, int y3, int w3, int h3, int x4, int y4, int w4, int h4, int t0, int t1, int &a,int &b,int &c,int &d)
{
	//����ͼ���и�ͼ��
	//Mat img = imread(srcPath, 1);
	//Mat img_model = imread(srcPath_model, 1);//��Ϊģ��ͼ��ɸѡ����ʱʹ�ã���洢·���̶�

	//Mat img_home = cutImage(img, x1, y1, w1, h1);

	//Mat img_syq = cutImage(img, x2, y2, w2, h2);

	//Mat img_xyq = cutImage(img, x3, y3, w3, h3);

	//Mat img_back = cutImage(img, x4, y4, w4, h4);
	////���ڲ���ƽ��������Ҫ����ͼƬ��С��ͬ����ˣ��˴�����ͼ���и��ʱ�������Ϊģ����ԭͼ�е�����



	//Mat img_home_model = cutImage(img_model, x1, y1, w1, h1);

	//Mat img_syq_model = cutImage(img_model, x2, y2, w2, h2);

	//Mat img_xyq_model = cutImage(img_model, x3, y3, w3, h3);

	//Mat img_back_model = cutImage(img_model, x4, y4, w4, h4);


	//rgbתΪhsi
	//Mat img_home_HSI;
	//Mat img_home_model_HSI;
	//Mat img_syq_model_HSI;
	//Mat img_xyq_model_HSI;
	//Mat img_back_model_HSI;
	//Mat img_syq_HSI;
	//Mat img_xyq_HSI;
	//Mat img_back_HSI;
	//cvtColor(img_home, img_home_HSI, COLOR_BGR2HLS);
	//cvtColor(img_syq, img_syq_HSI, COLOR_BGR2HLS);
	//cvtColor(img_xyq, img_xyq_HSI, COLOR_BGR2HLS);
	//cvtColor(img_back, img_back_HSI, COLOR_BGR2HLS);
	//cvtColor(img_home_model, img_home_model_HSI, COLOR_BGR2HLS);
	//cvtColor(img_syq_model, img_syq_model_HSI, COLOR_BGR2HLS);
	//cvtColor(img_xyq_model, img_xyq_model_HSI, COLOR_BGR2HLS);
	//cvtColor(img_back_model, img_back_model_HSI, COLOR_BGR2HLS);
	////��������
	//int avg_home = getEveryPixels(img_home_HSI, img_home_model_HSI, t0);
	//int avg_syq = getEveryPixels(img_syq_HSI, img_syq_model_HSI, t0);
	//int avg_xyq = getEveryPixels(img_xyq_HSI, img_xyq_model_HSI, t0);
	//int avg_back = getEveryPixels(img_back_HSI, img_back_model_HSI, t0);

	////����ƫ��ֵ��������
	//float avg = (avg_home + avg_syq + avg_xyq + avg_back) / 4;
	//if (avg < 10) {
	//	return 0;
	//}
	//float home = abs(avg_home - avg) / avg * 100;
	//float syq = abs(avg_syq - avg) / avg * 100;
	//float xyq = abs(avg_xyq - avg) / avg * 100;
	//float back = abs(avg_back - avg) / avg * 100;
	//if (home <= t1 && syq <= t1 && xyq <= t1 && back <= t1)
	//	return 1;//�ϸ�
	//else
	//	return 0;//���ϸ�

	Mat srcImg = imread(srcPath,1);
	Rect roi1(x1, y1, w1, h1);//home
	Rect roi2(x2, y2, w2, h2);//��һ��
	Rect roi3(x3, y3, w3, h3);//��һ��
	Rect roi4(x4, y4, w4, h4);//back

	Mat src1 = srcImg(roi1);
	Mat src2 = srcImg(roi2);
	Mat src3 = srcImg(roi3);
	Mat src4 = srcImg(roi4);
	// ת�ɻҶ�ͼ
	Mat srcGray1;
	Mat srcGray2;
	Mat srcGray3;
	Mat srcGray4;
	cvtColor(src1, srcGray1, COLOR_BGR2GRAY);
	cvtColor(src2, srcGray2, COLOR_BGR2GRAY);
	cvtColor(src3, srcGray3, COLOR_BGR2GRAY);
	cvtColor(src4, srcGray4, COLOR_BGR2GRAY);
	Mat Gaosi1;
	Mat Gaosi2;
	Mat Gaosi3;
	Mat Gaosi4;
	GaussianBlur(srcGray1, Gaosi1, Size(5, 5), 3);
	GaussianBlur(srcGray2, Gaosi2, Size(5, 5), 3);
	GaussianBlur(srcGray3, Gaosi3, Size(5, 5), 3);
	GaussianBlur(srcGray4, Gaosi4, Size(5, 5), 3);
	// ����Ӧ��ֵ��ֵ��
	Mat srcBinary1;
	Mat srcBinary2;
	Mat srcBinary3;
	Mat srcBinary4;
	adaptiveThreshold(~Gaosi1, srcBinary1, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 21, 3);
	adaptiveThreshold(~Gaosi2, srcBinary2, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 21, 3);
	adaptiveThreshold(~Gaosi3, srcBinary3, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 21, 3);
	adaptiveThreshold(~Gaosi4, srcBinary4, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 21, 3);
	Mat srcImg_out_withHSI1;
	Mat srcImg_out_withHSI2;
	Mat srcImg_out_withHSI3;
	Mat srcImg_out_withHSI4;
	cvtColor(src1, srcImg_out_withHSI1, CV_BGR2HLS);
	cvtColor(src2, srcImg_out_withHSI2, CV_BGR2HLS);
	cvtColor(src3, srcImg_out_withHSI3, CV_BGR2HLS);
	cvtColor(src4, srcImg_out_withHSI4, CV_BGR2HLS);

	int ans1 = getRes(srcBinary1, srcImg_out_withHSI1);
	int ans2 = getRes(srcBinary2, srcImg_out_withHSI2);
	int ans3 = getRes(srcBinary3, srcImg_out_withHSI3);
	int ans4 = getRes(srcBinary4, srcImg_out_withHSI4);

	float avg = (ans1 + ans2 + ans3 + ans4) / 4;
	if (avg < 5) {
		return 0;
	}
	float home = abs(ans1 - avg) / avg * 100;
	float syq = abs(ans2 - avg) / avg * 100;
	float xyq = abs(ans3 - avg) / avg * 100;
	float back = abs(ans4 - avg) / avg * 100;
	a = home;
	b = syq;
	c = xyq;
	d = back;
	if (home <= t1 && syq <= t1 && xyq <= t1 && back <= t1)
		return 1;//�ϸ�
	else
		return 0;//���ϸ�
}


/*
�������ԣ���Դ���ⲿԲ�����ȣ�
srcpath�������ͼ��
x1,y1,w1,h1:�ü��������꣬���Ͻ����꣬��ȣ����ȣ����ڵ�Դ�����Ϸ�Բ������
x2,y2,w2,h2:�ü��������꣬���Ͻ����꣬��ȣ����ȣ����ڵ�Դ�����Ϸ�Բ������
x3,y3,w3,h3:�ü��������꣬���Ͻ����꣬��ȣ����ȣ����ڵ�Դ����Բ������
x4,y4,w4,h4:�ü��������꣬���Ͻ����꣬��ȣ����ȣ����ڵ�Դ���ҷ�Բ������
x5,y5,w5,h5:�ü��������꣬���Ͻ����꣬��ȣ����ȣ����ڵ�Դ�����·�Բ������
x6,y6,w6,h6:�ü��������꣬���Ͻ����꣬��ȣ����ȣ����ڵ�Դ�����·�Բ������
t0������ɸѡ��ֵ
t1���ж���ֵ
*/
extern "C" PANELFUNDECTDLL_API int junyunxing(const char* srcPath, const char* srcPathmodel1, const char* srcPathmodel2, const char* srcPathmodel3, const char* srcPathmodel4, const char* srcPathmodel5, const char* srcPathmodel6, int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2, int x3, int y3, int w3, int h3, int x4, int y4, int w4, int h4, int x5, int y5, int w5, int h5, int x6, int y6, int w6, int h6, int t0, int t1)
{
	//����ͼ���и�
	Mat img = imread(srcPath, 1);
	//Mat img_model = imread(srcPathmodel, 1);
	//bright(img, 10.0, 5);
	Mat img_area1 = cutImage(img, x1, y1, w1, h1);
	Mat img_area2 = cutImage(img, x2, y2, w2, h2);
	Mat img_area3 = cutImage(img, x3, y3, w3, h3);
	Mat img_area4 = cutImage(img, x4, y4, w4, h4);
	Mat img_area5 = cutImage(img, x5, y5, w5, h5);
	Mat img_area6 = cutImage(img, x6, y6, w6, h6);
	Mat img_model1 = imread(srcPathmodel1, 1);
	Mat img_model2 = imread(srcPathmodel2, 1);
	Mat img_model3 = imread(srcPathmodel3, 1);
	Mat img_model4 = imread(srcPathmodel4, 1);
	Mat img_model5 = imread(srcPathmodel5, 1);
	Mat img_model6 = imread(srcPathmodel6, 1);

	//rgbתΪhsi
	Mat img_area1_HSI;
	Mat img_area2_HSI;
	Mat img_area3_HSI;
	Mat img_area4_HSI;
	Mat img_area5_HSI;
	Mat img_area6_HSI;
	Mat img_model1_HSI;
	Mat img_model2_HSI;
	Mat img_model3_HSI;
	Mat img_model4_HSI;
	Mat img_model5_HSI;
	Mat img_model6_HSI;
	cvtColor(img_area1, img_area1_HSI, COLOR_BGR2HLS);
	cvtColor(img_area2, img_area2_HSI, COLOR_BGR2HLS);
	cvtColor(img_area3, img_area3_HSI, COLOR_BGR2HLS);
	cvtColor(img_area4, img_area4_HSI, COLOR_BGR2HLS);
	cvtColor(img_area5, img_area5_HSI, COLOR_BGR2HLS);
	cvtColor(img_area6, img_area6_HSI, COLOR_BGR2HLS);
	cvtColor(img_model1, img_model1_HSI, COLOR_BGR2HLS);
	cvtColor(img_model2, img_model2_HSI, COLOR_BGR2HLS);
	cvtColor(img_model3, img_model3_HSI, COLOR_BGR2HLS);
	cvtColor(img_model4, img_model4_HSI, COLOR_BGR2HLS);
	cvtColor(img_model5, img_model5_HSI, COLOR_BGR2HLS);
	cvtColor(img_model6, img_model6_HSI, COLOR_BGR2HLS);

	//��ƽ������
	int avg_area1 = getEveryPixels(img_area1_HSI, img_model1_HSI, t0);
	int avg_area2 = getEveryPixels(img_area2_HSI, img_model2_HSI, t0);
	int avg_area3 = getEveryPixels(img_area3_HSI, img_model3_HSI, t0);
	int avg_area4 = getEveryPixels(img_area4_HSI, img_model4_HSI, t0);
	int avg_area5 = getEveryPixels(img_area5_HSI, img_model5_HSI, t0);
	int avg_area6 = getEveryPixels(img_area6_HSI, img_model6_HSI, t0);


	int t;
	t = 2 * t1;
	//����ƫ��ж�
	float avg = (avg_area1 + avg_area2 + avg_area3 + avg_area4 + avg_area5 + avg_area6) / 6;
	if (avg < 20) {
		return 0;
	}//���ȫ�ڣ���ֱ�ӱ���
	float area1 = abs(avg_area1 - avg) / avg * 100;
	float area2 = abs(avg_area2 - avg) / avg * 100;
	float area3 = abs(avg_area3 - avg) / avg * 100;
	float area4 = abs(avg_area4 - avg) / avg * 100;
	float area5 = abs(avg_area5 - avg) / avg * 100;
	float area6 = abs(avg_area6 - avg) / avg * 100;
	if (area1 <= t && area2 <= t && area3 <= t && area4 <= t && area5 <= t && area6 <= t)
		return 1;//ok
	else
		return 0;//ng

}

/*
��״���
srcPath:�����ͼ��
srcPathModel:˿ӡģ��ͼ��
x1,y1,w1,h1:˿ӡ�ü��������Ͻ����꣬��ȣ�����
t:���ƶ���ֵ
*/
extern "C" PANELFUNDECTDLL_API int siyinxingzhuang(const char* srcPath, const char* srcPathModel, int x1, int y1, int w1, int h1, double t,double &s)
{
	Mat img = imread(srcPath, 1);
	Mat imgModel = imread(srcPathModel, 1);
	Mat img_area1 = cutImage(img, x1, y1, w1, h1);

	Mat img_roi = getsquare(img_area1);
	if (img_roi.rows < 16 || img_roi.cols < 16)
	{
		s = 0;
		return 0;
	}
	 
	cvtColor(img_roi, img_roi, CV_BGR2GRAY);
	adaptiveThreshold(~img_roi, img_roi, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 17, 6);

	cvtColor(imgModel, imgModel, CV_BGR2GRAY);
	adaptiveThreshold(~imgModel, imgModel, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 17, 6);

	//�����е�ͼ����жԱ�ƥ��
	s = ImageCompared(img_roi, imgModel);
	if (s * 100 >= t)
		return 1;
	else
		return 0;
}

/*
λ�ü��
srcPath:�����ͼ��
x1,y1,w1,h1:˿ӡ�ü��������Ͻ����꣬��ȣ�����
t:˿ӡƫ�ƾ�����ֵ����λ������
a������������֮���ת����ϵ��1mm=a����
*/
extern "C" PANELFUNDECTDLL_API int siyinLocation(const char* srcPath, int x1, int y1, int w1, int h1, int x0, int y0, float t,int a, float &b)
{
    //��ȡͼ��
    Mat Image = imread(srcPath, 1);
    //����ͼ��
    Mat img = cutImage(Image, x1, y1, w1, h1);
    //������Ӿ��Σ��������׼λ�õľ���
    int distance = drawSquare(img, x0, y0);
	int t_pixel = a * t;  
	b = (float)distance / a;
	if (distance <= t_pixel)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}