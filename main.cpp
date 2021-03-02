#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include <vector>
#include <xmmintrin.h>
#include "opencv2/opencv.hpp"

class Timer {
    struct timespec s_;
public:
    Timer() { tic(); }
    void tic() {
        clock_gettime(CLOCK_REALTIME, &s_);
    }

    double toc() {
        struct timespec e;
        clock_gettime(CLOCK_REALTIME, &e);
        return (double)(e.tv_sec - s_.tv_sec) + 1e-9 * (double)(e.tv_nsec - s_.tv_nsec);
    }
};

void Intrinsic(float* MatrixA, float* MatrixB, float* MaxtrixDest)
{
	int indexA = 0;
	int indexB = 0;

	__m128 xmmB[4];
	int i = 0;
	for (i = 0; i < 4; i++)
		xmmB[i] = _mm_load_ps((MatrixB + (i * 4)));

	__m128 xmmR;
	for (i = 0; i < 4; i++)
	{
		indexB = 0;

		xmmR = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(MatrixA[indexA++]), xmmB[indexB++]),
				_mm_add_ps(_mm_mul_ps(_mm_set1_ps(MatrixA[indexA++]), xmmB[indexB++]),
				_mm_add_ps(_mm_mul_ps(_mm_set1_ps(MatrixA[indexA++]), xmmB[indexB++]),
				_mm_mul_ps(_mm_set1_ps(MatrixA[indexA++]), xmmB[indexB++]))));

		_mm_store_ps((MaxtrixDest + (i * 4)), xmmR);
	}
}

// Please optimize this function
void matmult(int m, int n, int k, const float* mat_a, const float* mat_b, float* mat_c)
{
    /*
        == input ==
        mat_a: m x k matrix
        mat_b: k x n matrix

        == output ==
        mat_c: m x n matrix (output)
    */

    for (int i1=0; i1<m; i1++) {
        for (int i2=0; i2<n; i2++) {
            mat_c [n*i1 + i2] = 0;
            for (int i3=0; i3<k; i3++) {
                mat_c[n*i1 + i2] += mat_a[i1 * k + i3] * mat_b[i3 * n + i2];
            }
        }
    }
}

void genmat(int n, int m, std::vector<float>& mat)
{
    srand(time(0));
    mat.resize(n * m);
    for (int i=0; i < mat.size(); i++) mat[i] = rand() % 100;
}

void dumpmat(int n, int m, std::vector<float>& mat)
{
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<m; j++)
            printf("%f ", mat[i * m + j]);
        printf("\n");
    }
}

void dumpmat(int n, int m, cv::Mat& mat)
{
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<m; j++)
            printf("%f ", mat.at<float>(i, j));
        printf("\n");
    }
}

int main(int argc, char** argv)
{
    std::vector<float> mat_a;
    std::vector<float> mat_b;
    std::vector<float> mat_c;


    genmat(4, 4, mat_a);
    genmat(4, 4, mat_b);
    genmat(4, 4, mat_c);
    
	Timer t;
    double elapsed=0;
    const int iteration = 10000;
    for (int i=0; i<iteration; i++)
    {
        t.tic();
        matmult(4, 4, 4, &mat_a[0], &mat_b[0], &mat_c[0]);
        elapsed += t.toc();
    }

	printf("\n<<< using loop >>>\n"); 
	printf("%lf ms\n", 1000.0 * elapsed / iteration);
    dumpmat(4, 4, mat_a);
    dumpmat(4, 4, mat_c);
	 

    elapsed=0;
    for (int i=0; i<iteration; i++)
    {
        t.tic();
        Intrinsic(&mat_a[0], &mat_b[0], &mat_c[0]);
        elapsed += t.toc();
    }

	printf("\n<<< using SIMD >>>\n"); 
    printf("%lf ms\n", 1000.0 * elapsed / iteration);
    dumpmat(4, 4, mat_a);
    dumpmat(4, 4, mat_c);

	// openCV	
	cv::Mat cv_a = cv::Mat(mat_a, false).reshape(1, 4);
	cv::Mat cv_b = cv::Mat(mat_b, false).reshape(1, 4);
	cv::Mat cv_c = cv::Mat(4, 4, CV_32F);

    elapsed=0;
    for (int i=0; i<iteration; i++)
    {
        t.tic();
		cv_c = cv_a*cv_b;
        elapsed += t.toc();
	} 	
	printf("\n<<< using openCV >>>\n"); 
    printf("%lf ms\n", 1000.0 * elapsed / iteration);
    dumpmat(4, 4, cv_a);
    dumpmat(4, 4, cv_c);
		
		
	return 0;
}
