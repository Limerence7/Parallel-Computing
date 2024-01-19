/* 
 * logDataVSPrior ��һ�����ڼ������鸴�����ݵľ���ֵ�ۼӵĺ���
 * *************************************************************************/

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>	// ʱ���ʱ��
#include <immintrin.h>
#include <xmmintrin.h>

using namespace std;

typedef chrono::high_resolution_clock Clock;

const int m=1638400;	// �������!!
const int K=100000;	// �������!!

// �������ֵ�ۼӵĺ���
float logDataVSPrior(const float* dat_real, const float* dat_comp, const float* pri_real, const float* pri_comp, const float* ctf, const float* sigRcp, const int num, const float disturb0);

int main ( int argc, char *argv[] )
{ 
    float *dat_real = new float[m]; // �洢�������ݵ�ʵ��
    float *dat_comp = new float[m]; // �洢�������ݵ��鲿
    float *pri_real = new float[m]; // �洢�������ݵ�ʵ��
    float *pri_comp = new float[m]; // �洢�������ݵ��鲿
    float *ctf = new float[m]; // �洢float���ݵ�����
    float *sigRcp = new float[m]; // �洢float���ݵ�����
    float *disturb = new float[K]; // �洢float���ݵ�����
    float dat0, dat1, pri0, pri1, ctf0, sigRcp0;
    
    /***************************
     * ��input.dat�ж�ȡ����
     * *************************/
    ifstream fin;

    fin.open("input.dat");
    if(!fin.is_open())
    {
        cout << "���ļ�input.datʱ����" << endl;
        exit(1);
    }
    int i=0;
    while( !fin.eof() ) 
    {
        fin >> dat0 >> dat1 >> pri0 >> pri1 >> ctf0 >> sigRcp0;
        dat_real[i] = dat0; // �洢ʵ��
        dat_comp[i] = dat1; // �洢�鲿
        pri_real[i] = pri0; // �洢ʵ��
        pri_comp[i] = pri1; // �洢�鲿
        ctf[i] = ctf0; // ���ļ��ж�ȡ�����ݴ洢��������
        sigRcp[i] = sigRcp0; // ���ļ��ж�ȡ�����ݴ洢��������
        i++;
        if(i == m) break;
    }
    fin.close();

    fin.open("K.dat");
    if(!fin.is_open())
    {
	    cout << "���ļ�K.datʱ����" << endl;
	    exit(1);
    }
    i=0;
    while( !fin.eof() )
    {
	    fin >> disturb[i];
	    i++;
	    if(i == K) break;
    }
    fin.close();

    /***************************
     * ��Ҫ���㲿��������
     * ************************/
    auto startTime = Clock::now(); 

    ofstream fout;
    fout.open("result.dat");
    if(!fout.is_open())
    {
         cout << "��result�ļ�ʱ����" << endl;
         exit(1);
    }

    for(unsigned int t = 0; t < K; t++)
    {
        float result = logDataVSPrior(dat_real, dat_comp, pri_real, pri_comp, ctf, sigRcp, m, disturb[t]);
        fout << t+1 << ": " << result << endl; // ��������д���ļ�
    }
    fout.close();

    auto endTime = Clock::now(); 

    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Computing time=" << compTime.count() << " microseconds" << endl;

    delete[] dat_real;
    delete[] dat_comp;
    delete[] pri_real;
    delete[] pri_comp;
    delete[] ctf;
    delete[] sigRcp;
    delete[] disturb;
    return EXIT_SUCCESS;
}


float logDataVSPrior(const float* dat_comp, const float* dat_real, const float* pri_comp, const float* pri_real, 
                const float* ctf, const float* sigRcp, const int num, const float disturb0) {
    float ans = 0.0;

    __m512 comp_avx, real_avx, result_avx = _mm512_setzero_ps();
    __m512 disturb0_avx = _mm512_set1_ps(disturb0);

    for (int i = 0; i < num; i += 16) {
        comp_avx = _mm512_sub_ps(_mm512_loadu_ps(&dat_comp[i]), 
                _mm512_mul_ps(_mm512_mul_ps(disturb0_avx, _mm512_loadu_ps(&ctf[i])), _mm512_loadu_ps(&pri_comp[i])));
        real_avx = _mm512_sub_ps(_mm512_loadu_ps(&dat_real[i]), 
                _mm512_mul_ps(_mm512_mul_ps(disturb0_avx, _mm512_loadu_ps(&ctf[i])), _mm512_loadu_ps(&pri_real[i])));
        
        result_avx = _mm512_add_ps(result_avx, _mm512_add_ps(_mm512_mul_ps(comp_avx, comp_avx), _mm512_mul_ps(real_avx, real_avx)));
    }

    float* result = new float[16];
    _mm512_store_ps(result, result_avx);


    for (int i = 0; i < 16; i++) {
        ans += result[i];
    }

    return ans;
}
