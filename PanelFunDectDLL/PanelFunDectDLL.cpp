// PanelFunDectDLL.cpp : 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"
#include "PanelFunDectDLL.h"

// 这是导出变量的一个示例
PANELFUNDECTDLL_API int nPanelFunDectDLL=0;

// 这是导出函数的一个示例。
 PANELFUNDECTDLL_API bool fnPanelFunDectDLL(void)
{
    return abs(1-9)>=100;
}

// 这是已导出类的构造函数。
// 有关类定义的信息，请参阅 PanelFunDectDLL.h
CPanelFunDectDLL::CPanelFunDectDLL()
{
    return;
}
/**
 遍历像素，将明度分量提高dev个数值
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
			// v Value 明度
			sum += inputImage.at<Vec3b>(i, j)[1];
		}
	}
	return sum / avg;

}

//四个点从左上角，右上角，左下，右下；w和h是校正后的图像宽和高；
/*
函数描述：校正图像
srcPath : 原图像路径
Point坐标 ： 原图像校正区域的四个点坐标；从左上角，右上角，左下，右下
width，height ： 校正后图像的宽和高；
*/
extern "C" PANELFUNDECTDLL_API int transformImg(const char * srcPath, const char * dstPath, float firstPointX, float firstPointY, float secondPointX, float secondPointY, float thridPointX, float thridPointY, float fourPointX, float fourPointY, float width, float height)
{
	//校正图像；
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
函数描述：屏幕亮度增加检测
srcPath : 校正后的原图
comparePath ： 校正后的当前图
x,y，w,h ： 裁剪区域坐标，左上角的坐标，宽度，长度；
threshold : 亮度差值的阈值
*/
extern "C" PANELFUNDECTDLL_API int brightPulsDetection(const char * srcPath,const char * comparePath,int x,int y,int w,int h,int threshold)
{
	 /*读取照片*/
	
	Mat srcImg = imread(srcPath);
	Mat compareImg = imread(comparePath);
	Rect roi(x, y, w, h);
	Mat srcImgCrop = srcImg(roi);
	Mat compareImgCrop = compareImg(roi);
	 //预定义另外色彩空间的变量
	Mat srcImg_out_withHSI;
	Mat compareImg_out_withHSI;
	// 转化色彩空间到HSV色彩空间
	cvtColor(srcImgCrop, srcImg_out_withHSI, CV_BGR2HLS);
	cvtColor(compareImgCrop, compareImg_out_withHSI, CV_BGR2HLS);
	// 将HSI色彩空间的明度
	int srcAvg = getEveryPixels(srcImg_out_withHSI);
	int compareAvg = getEveryPixels(compareImg_out_withHSI);   //利用实际采集图像与对比模板图像明度的差值阈值设置判断
	int difference = compareAvg - srcAvg;
	if (difference >= threshold) {
		return 1;
	}
	return 0;
	
}


//屏幕亮度-
/*
函数描述：屏幕亮度减少检测
srcPath : 校正后的参考图
comparePath ： 校正后的当前图
x,y，w,h ： 裁剪区域坐标，左上角的坐标，宽度，长度；
threshold : 亮度差值的阈值
*/
extern "C" PANELFUNDECTDLL_API int brightReduceDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int threshold)
{
	/*读取照片*/

	Mat srcImg = imread(srcPath, 1);
	Mat compareImg = imread(comparePath, 1);
	Rect roi(x, y, w, h);
	Mat srcImgCrop = srcImg(roi);
	Mat compareImgCrop = compareImg(roi);
	//预定义另外色彩空间的变量
	Mat srcImg_out_withHSI;
	Mat compareImg_out_withHSI;
	// 转化色彩空间到HSV色彩空间
	cvtColor(srcImgCrop, srcImg_out_withHSI, COLOR_BGR2HLS);
	cvtColor(compareImgCrop, compareImg_out_withHSI, COLOR_BGR2HLS);
	// 将HSI色彩空间的明度
	int srcAvg = getEveryPixels(srcImg_out_withHSI);
	int compareAvg = getEveryPixels(compareImg_out_withHSI);
	int difference = srcAvg - compareAvg;
	if (difference >= threshold) {
		return 1;
	}
	return 0;
}



//颜色检测，由绿变红；就是裁剪区域坐标不同
/*
函数描述：颜色检测
prePath : 校正后的参考图
srcPath ：校正后的当前图
x,y，w,h：裁剪区域坐标，左上角的坐标，宽度，长度；
greenThreshold : G通道的阈值当前为200；redThreshold:R通道的阈值值的设置为相应的实际情况而定
cntThreshold : 大于通道阈值的点个数,默认为3000（测量值0,7000）
*/
extern "C" PANELFUNDECTDLL_API int  colorDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold,int cntThreshold)
{
	Mat srcImg = imread(srcPath);
	Mat compareImg = imread(comparePath);
	Rect roi(x, y, w, h);              //Rect:设置一个矩形窗，roi为感兴趣的区域            
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
			//Vec3b代表长度为3的vector容器，数据类型为unchar 
			if (srcImgCrop.at<Vec3b>(i, j)[2] > redThreshold) {            //实际图像通过阈值的对应区域的像素点的数量
				srcCount++;
			}
			if (compareImgCrop.at<Vec3b>(i, j)[2] > redThreshold) {        //对比图像
				compareCount++;
			}
		}
	}
	if (abs(compareCount - srcCount) >= cntThreshold) {
		return 1;
	}
	return 0;
}

//HOME,上一曲，电源键，下一曲，back,音量+，音量-；就是裁剪区域坐标不同

/*
	功能描述：HOME检测
	prePath : 校正后的参考图
	srcPath ：校正后的当前图
	x,y,w,h ：裁剪区域坐标，左上角的坐标，宽度，长度；
	greenThreshold : G通道的阈值当前为100；
	cntThreshold : 大于通道阈值的点个数,默认为3000（测量值0,7000）
*/
extern "C" PANELFUNDECTDLL_API int homeDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold)
{
	return colorDetection(srcPath, comparePath, x,  y,  w,  h,  redThreshold, cntThreshold);
	
}
/*
功能描述：上一曲检测
prePath : 校正后的参考图
srcPath ：校正后的当前图
x,y,w,h ：裁剪区域坐标，左上角的坐标，宽度，长度；
greenThreshold : G通道的阈值当前为100；
cntThreshold : 大于通道阈值的点个数,默认为3000（测量值0,7000）
*/
extern "C" PANELFUNDECTDLL_API int songUpDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold)
{
	return colorDetection(srcPath, comparePath, x, y, w, h, redThreshold, cntThreshold);

}
/*
功能描述：下一曲检测
prePath : 校正后的参考图
srcPath ：校正后的当前图
x,y,w,h ：裁剪区域坐标，左上角的坐标，宽度，长度；
greenThreshold : G通道的阈值当前为100；
cntThreshold : 大于通道阈值的点个数,默认为3000（测量值0,7000）
*/
extern "C" PANELFUNDECTDLL_API int songDownDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold)
{
	return colorDetection(srcPath, comparePath, x, y, w, h, redThreshold, cntThreshold);

}
/*
功能描述：音量增加检测
prePath : 校正后的参考图
srcPath ：校正后的当前图
x,y,w,h ：裁剪区域坐标，左上角的坐标，宽度，长度；
greenThreshold : G通道的阈值当前为100；
cntThreshold : 大于通道阈值的点个数,默认为3000（测量值0,7000）
*/
extern "C" PANELFUNDECTDLL_API int volumeUpDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold)
{
	return colorDetection(srcPath, comparePath, x, y, w, h, redThreshold, cntThreshold);

}
/*
功能描述：音量减少检测
prePath : 校正后的参考图
srcPath ：校正后的当前图
x,y,w,h ：裁剪区域坐标，左上角的坐标，宽度，长度；
greenThreshold : G通道的阈值当前为100；
cntThreshold : 大于通道阈值的点个数,默认为3000（测量值0,7000）
*/
extern "C" PANELFUNDECTDLL_API int volumeDownDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold)
{
	return colorDetection(srcPath, comparePath, x, y, w, h, redThreshold, cntThreshold);

}


/*
功能描述：电源键检测
prePath : 校正后的参考图
srcPath ： 校正后的当前图
x,y,w,h ： 裁剪区域坐标，左上角的坐标，宽度，长度；
greenThreshold : G通道的阈值当前为100；
cntThreshold : 大于通道阈值的点个数,默认为3000（测量值0,7000）
*/
extern "C" PANELFUNDECTDLL_API int powerDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold)
{
	return colorDetection(srcPath, comparePath, x, y, w, h, redThreshold, cntThreshold);
}
/*
功能描述：back键检测
prePath : 校正后的参考图
srcPath ： 校正后的当前图
x,y,w,h ： 裁剪区域坐标，左上角的坐标，宽度，长度；
greenThreshold : G通道的阈值当前为100；
cntThreshold : 大于通道阈值的点个数,默认为3000（测量值0,7000）
*/
extern "C" PANELFUNDECTDLL_API int backDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold)
{
	return colorDetection(srcPath, comparePath, x, y, w, h, redThreshold, cntThreshold);
}
/*
功能描述：退出键检测
srcPath ： 校正后的当前图
x,y,w,h ： 裁剪区域坐标，左上角的坐标，宽度，长度；
greenThreshold : G通道的阈值；
cntThreshold : 大于通道阈值的点个数,
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
功能描述：打开测试软件检测
srcPath : 校正后的当前图
x,y,w,h ： 裁剪区域坐标，左上角的坐标，宽度，长度；
greenThreshold : G通道的
阈值；
cntThreshold : 小于通道阈值的点个数
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
函数描述：B通道颜色检测
prePath : 校正后的参考图
srcPath ： 校正后的当前图
x,y，w,h ： 裁剪区域坐标，左上角的坐标，宽度，长度；
greenThreshold : G通道的阈值当前为50；
cntThreshold : 大于通道阈值的点个数,默认为3000（测量值0,7778）
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
功能描述：背光键检测
prePath : 校正后的参考图
srcPath ： 校正后的当前图
x,y,w,h ： 裁剪区域坐标，左上角的坐标，宽度，长度；
greenThreshold : G通道的阈值当前为100；
cntThreshold : 大于通道阈值的点个数,默认为3000（测量值0,7000）
*/
extern "C" PANELFUNDECTDLL_API int  backLightDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int greenThreshold, int cntThreshold)
{
	return greenColorDetection(srcPath, comparePath, x, y, w, h, greenThreshold, cntThreshold);
}


/*
功能描述：数字识别检测
prePath : 校正后的参考图
srcPath ： 校正后的当前图
x,y,w,h ： 裁剪区域坐标，左上角的坐标，宽度，长度；
binThreshold : 二值化的阈值当前为25；
cntThreshold : 大于通道阈值的点个数,默认为3000（测量值0,7000）
*/
extern "C" PANELFUNDECTDLL_API int numRecognition(const char * srcPath, const char * numModelPath,int x, int y, int w, int h, int binThreshold)
{
	Mat srcImg = imread(srcPath);
	Rect roi(x, y, w, h);
	Mat srcImgCrop = srcImg(roi);
	if (!srcImgCrop.data) {
		return -1;
	}
	//对图像进行处理，转化为灰度图然后再转为二值图
	Mat grayImage;
	cvtColor(srcImgCrop, grayImage, CV_BGR2GRAY);
	Mat binImage;
	//第4个参数为CV_THRESH_BINARY_INV是因为我的输入原图为白底黑字
	//若为黑底白字则选择CV_THRESH_BINARY即可
	//0-5阈值为5,6和7的阈值为25
	//第一个参数（img）：源图像
	/* 第三个参数（255）：超过阈值的部分取值是多少（对于cv.THRESH_BINARY而言）
	第四个参数（cv.ADAPTIVE_THRESH_MEAN_C）：
	（1）在一个邻域内计算阈值所采用的算法，有两个取值，分别为 ADAPTIVE_THRESH_MEAN_C 和 ADAPTIVE_THRESH_GAUSSIAN_C
	（2）ADAPTIVE_THRESH_MEAN_C的计算方法是计算出领域的平均值再减去第七个参数2的值。
	（3）ADAPTIVE_THRESH_GAUSSIAN_C的计算方法是计算出领域的高斯均值再减去第七个参数2的值
	第五个参数（cv.THRESH_BINARY）：这是阈值类型，只有两个取值，分别为 THRESH_BINARY 和THRESH_BINARY_INV
	第六个参数（11）：adaptiveThreshold的计算单位是像素的邻域块大小选择，这是局部邻域大小，3、5、7等
	第七个参数（2）：这个参数实际上是一个偏移值调整量，用均值和高斯计算阈值后，再减或加这个值就是最终阈值。*/
	adaptiveThreshold(~grayImage, binImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 21, 4);   //自适应阈值，自适应阈值中每一个像素点的阈值是不同的，该阈值由其领域中图像像素带点加权平均决定。
	Mat binImage1 = binImage.clone();
	
	string numPath(numModelPath);
	numPath = numPath + "/%d.jpg";
	const char* model_filename = NULL;
	model_filename = numPath.c_str();
	//加载模板
	vector<Mat> myTemplate;
	for (int i = 0; i < 8; i++)
	{
		char name[128];
		sprintf_s(name, model_filename, i);
		Mat temp = imread(name, 0);
		myTemplate.push_back(temp);
	}
	// 按顺序取出和分割数字
	

	//进行比较,将图片与模板相减，然后求全部像素和，和最小表示越相似，进而完成匹配
	//MatchTemplate函数，是在一幅图像中寻找与另一幅模板图像最匹配(相似)部分。
	/*image：输入一个待匹配的图像，支持8U或者32F。
	  templ：输入一个模板图像，与image相同类型。
	  result：输出保存结果的矩阵，32F类型。
	  method：要使用的数据比较方法。*/
	vector<int> seq;  //顺序存放识别结果
	Mat res;
	double max = 0;
	int max_seq = 0;
	vector<double> ans;
	double a = 0;
	double b = 0;
	for (int j = 5; j < 8; j++)
	{
		matchTemplate(myTemplate[j], binImage, res, TM_CCOEFF_NORMED);  //一幅图像与另一幅图最佳匹配的地方
		//myTemplat：输入一个待匹配的图像，支持8U或者32F。8U指的是8位无符号数，32值得是32位的浮点数
	    //binImage：输入一个模板图像，与image相同类型。
	    //res：输出保存结果的矩阵，32F类型。
	    //TM_CCOEFF_NORMED：要使用的数据比较方法。
		
		minMaxLoc(res, &a, &b);  //寻找图像中最小最大值
		ans.push_back(b);
		if (b > max)            //比较所有匹配结果的相似度最大值
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


//亮度增强
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

//切割图像，image为待切割图像，xywd为切割坐标
Mat cutImage(Mat image, int x, int y, int w, int h)
{
	Mat img = image.clone();
	Rect rect(x, y, w, h);
	Mat ROI = img(rect);
	return ROI;
}
//测量平均亮度，inputImage为待检测图像，inputImage2为模板，t为亮度筛选阈值，
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
		return sum / num;      //返回一个平均亮度值
	}
}

/*
检测丝印亮度变化
srcPath：变化前图片
srcPath：变化后图片
x,y,w,h：裁剪区域坐标，左上角的坐标，宽度，长度
t0：亮度筛选阈值，默认为30
t1：亮度变化阈值
注：t1默认为正数
*/
extern "C" PANELFUNDECTDLL_API int liangdubianhua(const char* srcPath, const char* srcPath2, const char* srcPath3, int x, int y, int w, int h, int t0, int t1)
{
	//读取图像
	Mat img_before = imread(srcPath, 1);
	Mat img_after = imread(srcPath2, 1);
	Mat img_model = imread(srcPath2, 1);//此为模板图像，筛选亮度时使用，其存储路径固定
	
	//切割图像
	Mat img_cut_before = cutImage(img_before, x, y, w, h);
	Mat img_cut_after = cutImage(img_after, x, y, w, h);
	Mat img_cut_model = cutImage(img_model, x, y, w, h);

	bright(img_cut_before, 15.0, 5);
	bright(img_cut_after, 15.0, 5);
	bright(img_cut_model, 15.0, 5);

	//RGB转为HSI
	Mat img_before_HSI;
	Mat img_after_HSI;
	Mat img_model_HSI;
	cvtColor(img_cut_before, img_before_HSI, COLOR_BGR2HLS);
	cvtColor(img_cut_after, img_after_HSI, COLOR_BGR2HLS);
	cvtColor(img_cut_model, img_model_HSI, COLOR_BGR2HLS);

	//测量平均亮度
	int avg_before = getEveryPixels(img_before_HSI, img_model_HSI, t0);
	int avg_after = getEveryPixels(img_after_HSI, img_model_HSI, t0);

	//比较亮度差，输出结果，1为合格，0为不合格
	int difference = abs(avg_after - avg_before);   //操作的前后与模板进行对比得到差值
	if (difference >= t1 )
	{
		return 1;
	}
	else
	{
		return 0;
	}
}


//测量平均亮度，专用于一致性检测，image为待测图片，t为亮度筛选阈值
//注：
//由于一致性检测无需模板进行筛选像素点，因此本函数专用于一致性检测
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

//计算图片的指纹信息
string ImageHashValue(IplImage* src)
{
	string resStr(256, '\0');
	IplImage* image = cvCreateImage(cvGetSize(src), src->depth, 1);
	//step one : 灰度化
	if (src->nChannels == 3) cvCvtColor(src, image, CV_BGR2GRAY);
	else cvCopy(src, image);
	//step two : 缩小尺寸 16*16
	IplImage* temp = cvCreateImage(cvSize(16, 16), image->depth, 1);
	cvResize(image, temp);
	//step three : 简化色彩
	uchar* pData;
	for (int i = 0; i < temp->height; i++)
	{
		pData = (uchar*)(temp->imageData + i * temp->widthStep);
		for (int j = 0; j < temp->width; j++)
			pData[j] = pData[j] / 4;
	}
	//step four : 计算平均灰度值
	double average = cvAvg(temp).val[0];
	//step five : 计算哈希值
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

//根据指纹信息计算两幅图像的相似度
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

//相似度比较
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



//计算与标准位置的偏差距离，单位：像素
int drawSquare(Mat Image, int x0/*标准位置x坐标*/, int y0/*标准位置y坐标*/)
{
	Mat srcImg = Image.clone();
	Mat dstImg = srcImg.clone();
	Mat medianImg;
	medianBlur(srcImg, medianImg, 5);
	//imshow("中值滤波", medianImg);
	Mat brightImg = medianImg.clone();
	//bright(brightImg, 5.0, 5);
	//imshow("亮度增加", brightImg);

	cvtColor(brightImg, brightImg, CV_BGR2GRAY);
	Mat binaryImg;
	adaptiveThreshold(~brightImg, binaryImg, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 17, 6);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarcy;
	findContours(binaryImg, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<Rect> boundRect(contours.size());  //定义外接矩形集合

	vector<RotatedRect> box(contours.size()); //定义最小外接矩形集合
	Point2f rect[4];
	for (int i = 0; i < contours.size(); i++)
	{
		box[i] = minAreaRect(Mat(contours[i]));  //计算每个轮廓最小外接矩形
		boundRect[i] = boundingRect(Mat(contours[i]));
		//circle(dstImg, Point(box[i].center.x, box[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  //绘制最小外接矩形的中心点
		box[i].points(rect);  //把最小外接矩形四个端点复制给rect数组
		//rectangle(dstImg, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(255, 0, 0), 2, 8);
		for (int j = 0; j < 4; j++)
		{
			//line(dstImg, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //绘制最小外接矩形每条边
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


	vector<RotatedRect> box1(contours.size()); //定义最小外接矩形集合

	for (int i = 0; i < contours.size(); i++)
	{
		box1[i] = minAreaRect(Mat(contours[i]));  //计算每个轮廓最小外接矩形
		boundRect[i] = boundingRect(Mat(contours[i]));
		circle(dstImg, Point(box1[i].center.x, box1[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  //绘制最小外接矩形的中心点
		box1[i].points(rect);  //把最小外接矩形四个端点复制给rect数组
		rectangle(dstImg, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(255, 0, 0), 2, 8);
		for (int j = 0; j < 4; j++)
		{
			line(dstImg, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //绘制最小外接矩形每条边
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
	vector<Rect> boundRect(contours.size());  //定义外接矩形集合

	vector<RotatedRect> box(contours.size()); //定义最小外接矩形集合
	Point2f rect[4];
	for (int i = 0; i < contours.size(); i++)
	{
		box[i] = minAreaRect(Mat(contours[i]));  //计算每个轮廓最小外接矩形
		boundRect[i] = boundingRect(Mat(contours[i]));
		//circle(dstImg, Point(box[i].center.x, box[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  //绘制最小外接矩形的中心点
		box[i].points(rect);  //把最小外接矩形四个端点复制给rect数组
		//rectangle(dstImg, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(255, 0, 0), 2, 8);
		for (int j = 0; j < 4; j++)
		{
			//line(dstImg, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //绘制最小外接矩形每条边
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


	vector<RotatedRect> box1(contours.size()); //定义最小外接矩形集合

	for (int i = 0; i < contours.size(); i++)
	{
		box1[i] = minAreaRect(Mat(contours[i]));  //计算每个轮廓最小外接矩形
		boundRect[i] = boundingRect(Mat(contours[i]));
		circle(dstImg, Point(box1[i].center.x, box1[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  //绘制最小外接矩形的中心点
		box1[i].points(rect);  //把最小外接矩形四个端点复制给rect数组
		rectangle(dstImg, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(255, 0, 0), 2, 8);
		for (int j = 0; j < 4; j++)
		{
			line(dstImg, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //绘制最小外接矩形每条边
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

/*亮度一致性检测
srcPath：待检测图像
srcPath2：模板图像，用于计算亮度时筛选像素点
x1,y1,w1,h1:home键区域坐标，左上角坐标，宽度，长度
x1,y1,w1,h1:上一曲键区域坐标，左上角坐标，宽度，长度
x1,y1,w1,h1:下一曲键区域坐标，左上角坐标，宽度，长度
x1,y1,w1,h1:back键区域坐标，左上角坐标，宽度，长度
t0:亮度筛选阈值
t1:偏差百分比阈值
*/
extern "C" PANELFUNDECTDLL_API int yizhixing(const char* srcPath, const char* srcPath_model, int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2, int x3, int y3, int w3, int h3, int x4, int y4, int w4, int h4, int t0, int t1, int &a,int &b,int &c,int &d)
{
	//输入图像，切割图像
	//Mat img = imread(srcPath, 1);
	//Mat img_model = imread(srcPath_model, 1);//此为模板图像，筛选亮度时使用，其存储路径固定

	//Mat img_home = cutImage(img, x1, y1, w1, h1);

	//Mat img_syq = cutImage(img, x2, y2, w2, h2);

	//Mat img_xyq = cutImage(img, x3, y3, w3, h3);

	//Mat img_back = cutImage(img, x4, y4, w4, h4);
	////由于测量平均亮度需要两张图片大小相同，因此，此处进行图像切割的时候，坐标均为模板在原图中的坐标



	//Mat img_home_model = cutImage(img_model, x1, y1, w1, h1);

	//Mat img_syq_model = cutImage(img_model, x2, y2, w2, h2);

	//Mat img_xyq_model = cutImage(img_model, x3, y3, w3, h3);

	//Mat img_back_model = cutImage(img_model, x4, y4, w4, h4);


	//rgb转为hsi
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
	////测量亮度
	//int avg_home = getEveryPixels(img_home_HSI, img_home_model_HSI, t0);
	//int avg_syq = getEveryPixels(img_syq_HSI, img_syq_model_HSI, t0);
	//int avg_xyq = getEveryPixels(img_xyq_HSI, img_xyq_model_HSI, t0);
	//int avg_back = getEveryPixels(img_back_HSI, img_back_model_HSI, t0);

	////计算偏差值，输出结果
	//float avg = (avg_home + avg_syq + avg_xyq + avg_back) / 4;
	//if (avg < 10) {
	//	return 0;
	//}
	//float home = abs(avg_home - avg) / avg * 100;
	//float syq = abs(avg_syq - avg) / avg * 100;
	//float xyq = abs(avg_xyq - avg) / avg * 100;
	//float back = abs(avg_back - avg) / avg * 100;
	//if (home <= t1 && syq <= t1 && xyq <= t1 && back <= t1)
	//	return 1;//合格
	//else
	//	return 0;//不合格

	Mat srcImg = imread(srcPath,1);
	Rect roi1(x1, y1, w1, h1);//home
	Rect roi2(x2, y2, w2, h2);//上一曲
	Rect roi3(x3, y3, w3, h3);//下一曲
	Rect roi4(x4, y4, w4, h4);//back

	Mat src1 = srcImg(roi1);
	Mat src2 = srcImg(roi2);
	Mat src3 = srcImg(roi3);
	Mat src4 = srcImg(roi4);
	// 转成灰度图
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
	// 自适应阈值二值化
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
		return 1;//合格
	else
		return 0;//不合格
}


/*
检测均匀性（电源键外部圆环亮度）
srcpath：待检测图像
x1,y1,w1,h1:裁剪区域坐标，左上角坐标，宽度，长度，处于电源键左上方圆环区域
x2,y2,w2,h2:裁剪区域坐标，左上角坐标，宽度，长度，处于电源键右上方圆环区域
x3,y3,w3,h3:裁剪区域坐标，左上角坐标，宽度，长度，处于电源键左方圆环区域
x4,y4,w4,h4:裁剪区域坐标，左上角坐标，宽度，长度，处于电源键右方圆环区域
x5,y5,w5,h5:裁剪区域坐标，左上角坐标，宽度，长度，处于电源键左下方圆环区域
x6,y6,w6,h6:裁剪区域坐标，左上角坐标，宽度，长度，处于电源键右下方圆环区域
t0：亮度筛选阈值
t1：判断阈值
*/
extern "C" PANELFUNDECTDLL_API int junyunxing(const char* srcPath, const char* srcPathmodel1, const char* srcPathmodel2, const char* srcPathmodel3, const char* srcPathmodel4, const char* srcPathmodel5, const char* srcPathmodel6, int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2, int x3, int y3, int w3, int h3, int x4, int y4, int w4, int h4, int x5, int y5, int w5, int h5, int x6, int y6, int w6, int h6, int t0, int t1)
{
	//输入图像，切割
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

	//rgb转为hsi
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

	//测平均亮度
	int avg_area1 = getEveryPixels(img_area1_HSI, img_model1_HSI, t0);
	int avg_area2 = getEveryPixels(img_area2_HSI, img_model2_HSI, t0);
	int avg_area3 = getEveryPixels(img_area3_HSI, img_model3_HSI, t0);
	int avg_area4 = getEveryPixels(img_area4_HSI, img_model4_HSI, t0);
	int avg_area5 = getEveryPixels(img_area5_HSI, img_model5_HSI, t0);
	int avg_area6 = getEveryPixels(img_area6_HSI, img_model6_HSI, t0);


	int t;
	t = 2 * t1;
	//计算偏差并判断
	float avg = (avg_area1 + avg_area2 + avg_area3 + avg_area4 + avg_area5 + avg_area6) / 6;
	if (avg < 20) {
		return 0;
	}//如果全黑，则直接报错
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
形状检测
srcPath:待检测图像
srcPathModel:丝印模板图像
x1,y1,w1,h1:丝印裁剪区域，左上角坐标，宽度，长度
t:相似度阈值
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

	//将剪切的图像进行对比匹配
	s = ImageCompared(img_roi, imgModel);
	if (s * 100 >= t)
		return 1;
	else
		return 0;
}

/*
位置检测
srcPath:待检测图像
x1,y1,w1,h1:丝印裁剪区域，左上角坐标，宽度，长度
t:丝印偏移距离阈值，单位：毫米
a：毫米与像素之间的转化关系，1mm=a像素
*/
extern "C" PANELFUNDECTDLL_API int siyinLocation(const char* srcPath, int x1, int y1, int w1, int h1, int x0, int y0, float t,int a, float &b)
{
    //读取图像
    Mat Image = imread(srcPath, 1);
    //剪切图像
    Mat img = cutImage(Image, x1, y1, w1, h1);
    //绘制外接矩形，返回与标准位置的距离
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