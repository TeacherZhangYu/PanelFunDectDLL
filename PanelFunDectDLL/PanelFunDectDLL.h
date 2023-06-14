// 下列 ifdef 块是创建使从 DLL 导出更简单的
// 宏的标准方法。此 DLL 中的所有文件都是用命令行上定义的 PANELFUNDECTDLL_EXPORTS
// 符号编译的。在使用此 DLL 的
// 任何其他项目上不应定义此符号。这样，源文件中包含此文件的任何其他项目都会将
// PANELFUNDECTDLL_API 函数视为是从 DLL 导入的，而此 DLL 则将用此宏定义的
// 符号视为是被导出的。
#ifdef PANELFUNDECTDLL_EXPORTS
#define PANELFUNDECTDLL_API __declspec(dllexport)
#else
#define PANELFUNDECTDLL_API __declspec(dllimport)
#endif

#define _CRT_SECURE_NO_WARNINGS
#include "opencv/cv.h"
#include "opencv/highgui.h"  
#include "iostream"  
#include "math.h"  
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>

using namespace cv;
using namespace std;


// 此类是从 PanelFunDectDLL.dll 导出的
class PANELFUNDECTDLL_API CPanelFunDectDLL {
public:
	CPanelFunDectDLL(void);
	// TODO:  在此添加您的方法。
};

extern PANELFUNDECTDLL_API int nPanelFunDectDLL;

extern "C" PANELFUNDECTDLL_API bool fnPanelFunDectDLL(void);

//校正图像
extern "C" PANELFUNDECTDLL_API int transformImg(const char * srcPath, const char * dstPath, float firstPointX, float firstPointY, float secondPointX, float secondPointY, float thridPointX, float thridPointY, float fourPointX, float fourPointY, float width, float height);
//亮度增加检测
extern "C" PANELFUNDECTDLL_API int brightPulsDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int threshold);
//亮度减少检测
extern "C" PANELFUNDECTDLL_API int brightReduceDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int threshold);
//Home键检测
extern "C" PANELFUNDECTDLL_API int homeDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold);
//上一曲检测
extern "C" PANELFUNDECTDLL_API int songUpDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold);
//下一曲检测
extern "C" PANELFUNDECTDLL_API int songDownDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold);
//音量增加检测
extern "C" PANELFUNDECTDLL_API int volumeUpDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold);
//音量减少检测
extern "C" PANELFUNDECTDLL_API int volumeDownDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold);
//电源键检测
extern "C" PANELFUNDECTDLL_API int powerDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold);
//back键检测
extern "C" PANELFUNDECTDLL_API int backDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold);
//退出键检测
extern "C" PANELFUNDECTDLL_API int exitDetection(const char * srcPath, int x, int y, int w, int h, int redThreshold, int cntThreshold);
//安卓测试软件打开检测
extern "C" PANELFUNDECTDLL_API int openAppDetection(const char * srcPath, int x, int y, int w, int h, int redThreshold, int cntThreshold);
//关闭背光检测
extern "C" PANELFUNDECTDLL_API int backLightDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int greenThreshold, int cntThreshold);
//数字识别
extern "C" PANELFUNDECTDLL_API int numRecognition(const char * srcPath, const char * numModelPath, int x, int y, int w, int h, int binThreshold);



extern "C" PANELFUNDECTDLL_API int liangdubianhua(const char* srcPath, const char* srcPath2, const char* srcPath3, int x, int y, int w, int h, int t0, int t1);
//检测丝印亮度变化
extern "C" PANELFUNDECTDLL_API int yizhixing(const char* srcPath, const char* srcPath_model, int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2, int x3, int y3, int w3, int h3, int x4, int y4, int w4, int h4, int t0, int t1,int &a,int &b,int &c ,int &d);
//检测丝印一致性
extern "C" PANELFUNDECTDLL_API int junyunxing(const char* srcPath, const char* srcPathmodel1, const char* srcPathmodel2, const char* srcPathmodel3, const char* srcPathmodel4, const char* srcPathmodel5, const char* srcPathmodel6, int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2, int x3, int y3, int w3, int h3, int x4, int y4, int w4, int h4, int x5, int y5, int w5, int h5, int x6, int y6, int w6, int h6, int t0, int t1);
//检测丝印均匀性
extern "C" PANELFUNDECTDLL_API int siyinxingzhuang(const char* srcPath, const char* srcPathModel, int x1, int y1, int w1, int h1, double t, double &s);
//检测丝印形状
extern "C" PANELFUNDECTDLL_API int siyinLocation(const char* srcPath, int x1, int y1, int w1, int h1, int x0, int y0, float t,int a, float &b);
//丝印位置检测