// ���� ifdef ���Ǵ���ʹ�� DLL �������򵥵�
// ��ı�׼�������� DLL �е������ļ��������������϶���� PANELFUNDECTDLL_EXPORTS
// ���ű���ġ���ʹ�ô� DLL ��
// �κ�������Ŀ�ϲ�Ӧ����˷��š�������Դ�ļ��а������ļ����κ�������Ŀ���Ὣ
// PANELFUNDECTDLL_API ������Ϊ�Ǵ� DLL ����ģ����� DLL ���ô˺궨���
// ������Ϊ�Ǳ������ġ�
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


// �����Ǵ� PanelFunDectDLL.dll ������
class PANELFUNDECTDLL_API CPanelFunDectDLL {
public:
	CPanelFunDectDLL(void);
	// TODO:  �ڴ�������ķ�����
};

extern PANELFUNDECTDLL_API int nPanelFunDectDLL;

extern "C" PANELFUNDECTDLL_API bool fnPanelFunDectDLL(void);

//У��ͼ��
extern "C" PANELFUNDECTDLL_API int transformImg(const char * srcPath, const char * dstPath, float firstPointX, float firstPointY, float secondPointX, float secondPointY, float thridPointX, float thridPointY, float fourPointX, float fourPointY, float width, float height);
//�������Ӽ��
extern "C" PANELFUNDECTDLL_API int brightPulsDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int threshold);
//���ȼ��ټ��
extern "C" PANELFUNDECTDLL_API int brightReduceDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int threshold);
//Home�����
extern "C" PANELFUNDECTDLL_API int homeDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold);
//��һ�����
extern "C" PANELFUNDECTDLL_API int songUpDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold);
//��һ�����
extern "C" PANELFUNDECTDLL_API int songDownDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold);
//�������Ӽ��
extern "C" PANELFUNDECTDLL_API int volumeUpDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold);
//�������ټ��
extern "C" PANELFUNDECTDLL_API int volumeDownDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold);
//��Դ�����
extern "C" PANELFUNDECTDLL_API int powerDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold);
//back�����
extern "C" PANELFUNDECTDLL_API int backDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int redThreshold, int cntThreshold);
//�˳������
extern "C" PANELFUNDECTDLL_API int exitDetection(const char * srcPath, int x, int y, int w, int h, int redThreshold, int cntThreshold);
//��׿��������򿪼��
extern "C" PANELFUNDECTDLL_API int openAppDetection(const char * srcPath, int x, int y, int w, int h, int redThreshold, int cntThreshold);
//�رձ�����
extern "C" PANELFUNDECTDLL_API int backLightDetection(const char * srcPath, const char * comparePath, int x, int y, int w, int h, int greenThreshold, int cntThreshold);
//����ʶ��
extern "C" PANELFUNDECTDLL_API int numRecognition(const char * srcPath, const char * numModelPath, int x, int y, int w, int h, int binThreshold);



extern "C" PANELFUNDECTDLL_API int liangdubianhua(const char* srcPath, const char* srcPath2, const char* srcPath3, int x, int y, int w, int h, int t0, int t1);
//���˿ӡ���ȱ仯
extern "C" PANELFUNDECTDLL_API int yizhixing(const char* srcPath, const char* srcPath_model, int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2, int x3, int y3, int w3, int h3, int x4, int y4, int w4, int h4, int t0, int t1,int &a,int &b,int &c ,int &d);
//���˿ӡһ����
extern "C" PANELFUNDECTDLL_API int junyunxing(const char* srcPath, const char* srcPathmodel1, const char* srcPathmodel2, const char* srcPathmodel3, const char* srcPathmodel4, const char* srcPathmodel5, const char* srcPathmodel6, int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2, int x3, int y3, int w3, int h3, int x4, int y4, int w4, int h4, int x5, int y5, int w5, int h5, int x6, int y6, int w6, int h6, int t0, int t1);
//���˿ӡ������
extern "C" PANELFUNDECTDLL_API int siyinxingzhuang(const char* srcPath, const char* srcPathModel, int x1, int y1, int w1, int h1, double t, double &s);
//���˿ӡ��״
extern "C" PANELFUNDECTDLL_API int siyinLocation(const char* srcPath, int x1, int y1, int w1, int h1, int x0, int y0, float t,int a, float &b);
//˿ӡλ�ü��