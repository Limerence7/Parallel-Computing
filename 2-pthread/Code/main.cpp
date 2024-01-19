/* 
 * logDataVSPrior is a function to calculate 
 * the accumulation from ABS of two groups of complex data
 * *************************************************************************/

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include <pthread.h>

using namespace std;

typedef complex<double> Complex;
typedef chrono::high_resolution_clock Clock;

const int m=1638400;    // DO NOT CHANGE!!
const int K=100000; // DO NOT CHANGE!!

struct thread_Info {
    const float* dat_comp;
    const float* dat_real;
    const float* pri_comp;
    const float* pri_real;
    const float* ctf;
    const float* sigRcp;
    float disturb0, result;
    int len;
};

int cur_thread_num = 0;
const int Total_Threads = 8;
pthread_t* threads = new pthread_t[Total_Threads];
float* answers = new float[K];

void* logDataVSPrior(void* arg) {
    thread_Info* ti = (thread_Info*)arg;
    for (int i = 0; i < ti->len; i++) 
    {
        float comp = ti->dat_comp[i] - ti->disturb0 * ti->ctf[i] * ti->pri_comp[i];
        float real = ti->dat_real[i] - ti->disturb0 * ti->ctf[i] * ti->pri_real[i];
        
        ti->result += (comp * comp + real * real) * ti->sigRcp[i];
    }
    pthread_exit(NULL);
}

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

    for (unsigned int t = 0; t < K; t++) {
        thread_Info ti;
        ti.dat_comp = dat_comp;
        ti.dat_real = dat_real;
        ti.pri_comp = pri_comp;
        ti.pri_real = pri_real;
        
        ti.ctf = ctf;
        ti.sigRcp = sigRcp;
        ti.len = m;
        ti.disturb0 = disturb[t];
        ti.result = 0.0;

        pthread_create(&threads[cur_thread_num], NULL, logDataVSPrior, (void*)&ti);
        cur_thread_num = (cur_thread_num + 1) % Total_Threads;

        if (cur_thread_num == 0 || t == K - 1) {
            for (int i = 0; i <= cur_thread_num; i++) {
                pthread_join(threads[i], NULL);
                answers[t - cur_thread_num + i] = ti.result;
            }
        }
    }

    for (unsigned int t = 0; t < K; t++) {
        fout << t + 1 << ": " << answers[t] << endl;
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
