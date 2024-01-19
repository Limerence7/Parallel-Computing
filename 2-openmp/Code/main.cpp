/* 
 * logDataVSPrior is a function to calculate 
 * the accumulation from ABS of two groups of complex data
 * *************************************************************************/

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include <omp.h>

using namespace std;

typedef complex<double> Complex;
typedef chrono::high_resolution_clock Clock;

const int m=1638400;    // DO NOT CHANGE!!
const int K=100000; // DO NOT CHANGE!!

float logDataVSPrior(const float* dat_comp, const float* dat_real, const float* pri_comp, const float* pri_real, 
                const float* ctf, const float* sigRcp, const int num, const float disturb0);

int main ( int argc, char *argv[] )
{
    float *dat_real = new float[m];
    float *dat_comp = new float[m]; 
    float *pri_real = new float[m]; 
    float *pri_comp = new float[m]; 
    float *ctf = new float[m]; 
    float *sigRcp = new float[m]; 
    float *disturb = new float[K]; 
    float dat0, dat1, pri0, pri1, ctf0, sigRcp0;

    /***************************
     * Read data from input.dat
     * *************************/
    ifstream fin;

    fin.open("input.dat");
    if(!fin.is_open())
    {
        cout << "Error opening file input.dat" << endl;
        exit(1);
    }
    int i=0;
    while( !fin.eof() ) 
    {
        fin >> dat0 >> dat1 >> pri0 >> pri1 >> ctf0 >> sigRcp0;
        dat_comp[i] = dat0; 
        dat_real[i] = dat1;
        pri_comp[i] = pri0;
        pri_real[i] = pri1;
        ctf[i] = ctf0;
        sigRcp[i] = sigRcp0;
        i++;
        if(i == m) break;
    }
    fin.close();

    fin.open("K.dat");
    if(!fin.is_open())
    {
    cout << "Error opening file K.dat" << endl;
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
     * main computation is here
     * ************************/
    auto startTime = Clock::now(); 

    ofstream fout;
    fout.open("result.dat");
    if(!fout.is_open())
    {
         cout << "Error opening file for result" << endl;
         exit(1);
    }

    for(unsigned int t = 0; t < K; t++)
    {
        float result = logDataVSPrior(dat_comp, dat_real, pri_comp, pri_real, ctf, sigRcp, m, disturb[t]);
        fout << t+1 << ": " << result << endl;
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
                const float* ctf, const float* sigRcp, const int num, const float disturb0)
{
    float result = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:result)
    for (int i = 0; i < num; i++)
    {
        float comp = dat_comp[i] - disturb0 * ctf[i] * pri_comp[i];
        float real = dat_real[i] - disturb0 * ctf[i] * pri_real[i];
        
        result += (real * real + comp * comp) * sigRcp[i];

    }
    return result;
}