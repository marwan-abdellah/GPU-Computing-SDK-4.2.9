/*
* Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

#include <npp.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>

#include <iostream>
#include <fstream>

#include <string.h>
#include <assert.h>
#include <stdio.h>

#include <shrQATest.h>

#ifdef WIN32
#define STRCASECMP  _stricmp
#define STRNCASECMP _strnicmp
#else
#define STRCASECMP  strcasecmp
#define STRNCASECMP strncasecmp
#endif

// This is the CUDA 4.1/4.2 function naming
#define NPP_CUDA_VERSION ((NPP_VERSION_MAJOR << 12) | (NPP_VERSION_MINOR << 4))

bool g_bQATest = false;
int  g_nDevice = -1;

#ifndef WIN32

#define fscanf_s fscanf

inline void fopen_s(FILE** file, const char* name, const char* mode) 
{
    *file = fopen(name, mode);
}
#endif

//////////////////////////////////////////////////////////////////////////////
//! Find the path for a file assuming that
//! files are either in data/ or in ../../../src/<executable_name>/data/.
//!
//! @return the path if succeeded, otherwise 0
//! @param filename         name of the file
//! @param executable_path  optional absolute path of the executable
//////////////////////////////////////////////////////////////////////////////
char*
findFilePath(const char* filename, const char* executable_path) 
{
    // search in data/
    if (filename == 0)
        return 0;
    size_t filename_len = strlen(filename);
    const char data_folder[] = "../../data/";
    size_t data_folder_len = strlen(data_folder);
    char* file_path = 
      (char*) malloc( sizeof(char) * (data_folder_len + filename_len + 1));
    strcpy(file_path, data_folder);
    strcat(file_path, filename);
	size_t file_path_len = strlen(file_path);
	file_path[file_path_len] = '\0';
    std::fstream fh0(file_path, std::fstream::in);
    if (fh0.good())
        return file_path;
    free( file_path);

    // search in ../../../src/<executable_name>/data/
    if (executable_path == 0)
        return 0;
    size_t executable_path_len = strlen(executable_path);
    const char* exe;
    for (exe = executable_path + executable_path_len - 1; 
         exe >= executable_path; --exe)
        if (*exe == '/' || *exe == '\\')
            break;
    if (exe < executable_path)
        exe = executable_path;
    else
        ++exe;
    size_t executable_len = strlen(exe);
    size_t executable_dir_len = executable_path_len - executable_len;
    const char projects_relative_path[] = "../../../src/";
    size_t projects_relative_path_len = strlen(projects_relative_path);
    file_path = 
      (char*) malloc( sizeof(char) * (executable_path_len +
         projects_relative_path_len + 1 + data_folder_len + filename_len + 1));
    strncpy(file_path, executable_path, executable_dir_len);
    file_path[executable_dir_len] = '\0';
    strcat(file_path, projects_relative_path);
    strcat(file_path, exe);
    file_path_len = strlen(file_path);
    if (*(file_path + file_path_len - 1) == 'e' &&
        *(file_path + file_path_len - 2) == 'x' &&
        *(file_path + file_path_len - 3) == 'e' &&
        *(file_path + file_path_len - 4) == '.') {
        *(file_path + file_path_len - 4) = '/';
        *(file_path + file_path_len - 3) = '\0';
    }
    else {
        *(file_path + file_path_len - 0) = '/';
        *(file_path + file_path_len + 1) = '\0';
    }
    strcat(file_path, data_folder);
    strcat(file_path, filename);
	file_path_len = strlen(file_path);
	file_path[file_path_len] = '\0';
	std::fstream fh1(file_path, std::fstream::in);
    if (fh1.good())
        return file_path;
    free( file_path);
    return 0;
}


void loadMiddleburyMRFData(const std::string &filename, int* &dataCostArray, int* &hCueTransposed, int* &vCue, int &width, int &height, int &nLabels) 
{
	FILE *fp;
	fopen_s(&fp, filename.c_str(),"rb");
    
	if( fp == 0 ) throw(new npp::Exception("File not found!"));

	fscanf_s(fp,"%d %d %d",&width,&height,&nLabels);

	int i, n, x, y;
	int gt;
	for(i = 0; i < width * height; i++)
		fscanf_s(fp,"%d",&gt);

	dataCostArray = (int*) malloc(width * height * nLabels * sizeof(int));
    n = 0;
	int v;

	for(int c=0; c < nLabels; c++) {
		for(i = 0; i < width * height; i++) {
			fscanf_s(fp,"%d",&v);
			dataCostArray[n++] = v;
		}
	}

	hCueTransposed = (int*) malloc(width * height * sizeof(int));
	vCue = (int*) malloc(width * height * sizeof(int));

	for(y = 0; y < height; y++) {
		for(x = 0; x < width-1; x++) {
			fscanf_s(fp,"%d",&v);
			hCueTransposed[x*height+y] = v;
		}
		hCueTransposed[(width-1)*height+y] = 0;
	}

	for(y = 0; y < height-1; y++) {
		for(x = 0; x < width; x++) {
			fscanf_s(fp,"%d",&v);
			vCue[y*width+x] = v;
		}
	}
	for(x = 0; x < width; x++) {
		vCue[(height-1)*width+x] = 0;
	}

	fclose(fp);

}


int computeEnergy(int &E_d, int &E_s, unsigned char* h_label, int step, int* hCueTransposed, int* vCue, int* data, int width, int height)
{
    E_d = 0;
    E_s = 0;

    for( int y=0; y<height; ++y) {
        for( int x=0; x < width; ++x) {
            int myLabel = h_label[y*step+x] > 0 ? 1 : 0;
            
            // Data
            E_d += data[myLabel * width*height + y*width+x];
            

            // Right
            if( x< width-1 ) {
                int rightLabel = h_label[y*width+x+1] > 0 ? 1 : 0;
                if( rightLabel != myLabel ) {
                    E_s += hCueTransposed[x*height+y];
                }
            }

            // Bottom
            if( y < height-1 ) {
                int bottomLabel = h_label[(y+1)*width+x] > 0 ? 1 : 0;
                if( bottomLabel != myLabel ) {
                    E_s += vCue[y*width+x];
                }
            }

        }
    }

	return E_d + E_s;
}

inline void cudaSafeCallNoSync( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
        std::cerr << file << "(" << line << ")" << " : cudaSafeCallNoSync() Runtime API error : ";
	std::cerr << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

inline int cudaDeviceInit()
{
    int deviceCount;
    cudaSafeCallNoSync(cudaGetDeviceCount(&deviceCount), __FILE__, __LINE__);
    if (deviceCount == 0) {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(-1);
    }
    int dev = g_nDevice;
    if (dev < 0) 
        dev = 0;
    if (dev > deviceCount-1) {
        std::cerr << std::endl << ">> %d CUDA capable GPU device(s) detected. <<" << deviceCount << std::endl;
        std::cerr <<">> cutilDeviceInit (-device=" << dev << ") is not a valid GPU device. <<" << std::endl << std::endl;
        return -dev;
    }  else {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl; 
    }
    cudaSafeCallNoSync(cudaSetDevice(dev), __FILE__, __LINE__);

    return dev;
}

void parseCommandLineArguments(int argc, char *argv[])
{
	if (argc >= 2) {
        for (int i=1; i < argc; i++) {

            if (!STRCASECMP(argv[i], "-qatest")   || !STRCASECMP(argv[i], "--qatest") ||
				!STRCASECMP(argv[i], "-noprompt") || !STRCASECMP(argv[i], "--noprompt")) 
			{
				g_bQATest = true;
			}

			if (!STRNCASECMP(argv[i], "-device", 7)) {
				g_nDevice = atoi(&argv[i][8]);
			} else if (!STRNCASECMP(argv[i], "--device", 8)) {
				g_nDevice = atoi(&argv[i][9]);
		    }
			if (g_nDevice != -1) {
				cudaDeviceInit();
			}
		}
	}
}

void printfNPPinfo(int argc, char *argv[])
{
    const char *sComputeCap[] = {
       "No CUDA Capable Device Found",
       "Compute 1.0", "Compute 1.1", "Compute 1.2",  "Compute 1.3",
       "Compute 2.0", "Compute 2.1", "Compute 3.0", NULL
    };

    const NppLibraryVersion * libVer   = nppGetLibVersion();
    NppGpuComputeCapability computeCap = nppGetGpuComputeCapability();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);
    if (computeCap != 0 && g_nDevice == -1) {
        printf("%s using GPU <%s> with %d SM(s) with", argv[0], nppGetGpuName(), nppGetGpuNumSMs());
	if (computeCap > 0) { 
	    printf(" %s\n", sComputeCap[computeCap]);
	} else {
	    printf(" Unknown Compute Capabilities\n");
	}
    } else {
        printf("%s\n", sComputeCap[computeCap]);
    }
}

int main(int argc, char* argv[])
{
    shrQAStart(argc, argv);

    try
    {
		std::string sFilename;
		char *filePath = findFilePath("person.txt", argv[0]);
		if (filePath) {
			sFilename = filePath;
		} else {
			printf("Error %s was unable to find person.txt\n", argv[0]);
            shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
		}

		// Parse the command line arguments for proper configuration
		parseCommandLineArguments(argc, argv);

        printfNPPinfo(argc, argv);

        if (g_bQATest == false && (g_nDevice == -1) && argc > 1) {
			sFilename = argv[1];
		}

        // if we specify the filename at the command line, then we only test sFilename
        int file_errors = 0;

	    std::ifstream infile(sFilename.data(), std::ifstream::in);
        if (infile.good()) {
            std::cout << "imageSegmentationNPP opened: <" << sFilename.data() << "> successfully!" << std::endl;
            file_errors = 0;
		    infile.close();
        } else {
            std::cout << "imageSegmentationNPP unable to open: <" << sFilename.data() << ">" << std::endl;
            file_errors++;
		    infile.close();
        }

		if (file_errors > 0) {
            shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
        }

        std::string sResultFilename = sFilename;
        
        std::string::size_type dot = sResultFilename.rfind('.');
        if (dot != std::string::npos) sResultFilename = sResultFilename.substr(0, dot);
        sResultFilename += "_segmentation.pgm";

        if (argc >= 3 && !g_bQATest)
            sResultFilename = argv[2];

		// load MRF declaration
		int width, height, nLabels;
		int *hCue, *vCue, *dataCostArray;

		loadMiddleburyMRFData(sFilename, dataCostArray, hCue, vCue, width, height, nLabels);
		NPP_ASSERT(nLabels == 2);

		std::cout << "Dataset: " << sFilename << std::endl;
		std::cout << "Size: " << width << "x" << height << std::endl;

		NppiSize size;
		size.width = width;
		size.height = height;

		NppiRect roi;
		roi.x=0; roi.y=0;
		roi.width=width; roi.height=height;

		// Setup flow network
		int step, transposed_step;
		Npp32s *d_source, *d_sink, *d_terminals, *d_left_transposed, *d_right_transposed, *d_top, *d_bottom;

		// Setup terminal capacities
		d_source = nppiMalloc_32s_C1(width, height, &step);
		cudaMemcpy2D(d_source, step, dataCostArray, width * sizeof(int), width*sizeof(int), height, cudaMemcpyHostToDevice);
		d_sink = nppiMalloc_32s_C1(width, height, &step);
		cudaMemcpy2D(d_sink, step, &dataCostArray[width*height], width * sizeof(int), width*sizeof(int), height, cudaMemcpyHostToDevice);

		d_terminals = nppiMalloc_32s_C1(width, height, &step);

#if (NPP_CUDA_VERSION >= 0x4010)
        nppiSub_32s_C1RSfs(d_sink, step, d_source, step, d_terminals, step, size, 0);
#else
        nppiSub_32s_C1R(d_sink, step, d_source, step, d_terminals, step, size);
#endif

		// Setup edge capacities
		NppiSize edgeTranposedSize;
		edgeTranposedSize.width = height;
		edgeTranposedSize.height = width-1;

		NppiSize oneRowTranposedSize;
		oneRowTranposedSize.width = height;
		oneRowTranposedSize.height = 1;

		d_right_transposed = nppiMalloc_32s_C1(height, width, &transposed_step);
		cudaMemcpy2D(d_right_transposed, transposed_step, hCue, height * sizeof(int), height * sizeof(int), width, cudaMemcpyHostToDevice);
		
		d_left_transposed = nppiMalloc_32s_C1(height, width, &transposed_step);
		nppiSet_32s_C1R(0, d_left_transposed, transposed_step, oneRowTranposedSize);
		nppiCopy_32s_C1R(d_right_transposed, transposed_step, d_left_transposed + transposed_step/sizeof(int), transposed_step, edgeTranposedSize);

		NppiSize edgeSize;
		edgeSize.width = width;
		edgeSize.height = height-1;

		NppiSize oneRowSize;
		oneRowSize.width = width;
		oneRowSize.height = 1;

		d_bottom = nppiMalloc_32s_C1(width, height, &step);
		cudaMemcpy2D(d_bottom, step, vCue, width * sizeof(int), width*sizeof(int), height, cudaMemcpyHostToDevice);
		
		d_top = nppiMalloc_32s_C1(width, height, &step);
		nppiSet_32s_C1R(0, d_top, step, oneRowSize);
		nppiCopy_32s_C1R(d_bottom, step, d_top + step/sizeof(int), step, edgeSize);
		
		// Allocate temp storage for graphcut computation
		Npp8u* pBuffer;
		int bufferSize;
		nppiGraphcutGetSize(size, &bufferSize);
		cudaMalloc(&pBuffer, bufferSize);

#if (NPP_CUDA_VERSION >= 0x4010)
        NppiGraphcutState* pGraphcutState;
		nppiGraphcutInitAlloc(size, &pGraphcutState, pBuffer);	
#endif
		
		// Allocate label storage
		npp::ImageNPP_8u_C1 oDeviceDst(width, height);

		cudaEvent_t start, stop;
		cudaEventCreate(&start); cudaEventCreate(&stop);

		// Compute the graphcut
		cudaEventRecord(start,0);

#if (NPP_CUDA_VERSION >= 0x4010)
        nppiGraphcut_32s8u(d_terminals, d_left_transposed, d_right_transposed, 
                           d_top, d_bottom, step, transposed_step, 
                           size, oDeviceDst.data(), oDeviceDst.pitch(), pGraphcutState);
#else
        nppiGraphcut_32s8u(d_terminals, d_left_transposed, d_right_transposed, d_top, d_bottom, step, transposed_step, size, oDeviceDst.data(),
			oDeviceDst.pitch(), pBuffer);
#endif

		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);

		float time;
		cudaEventElapsedTime(&time, start, stop);
		std::cout << "Elapsed Time: " << time << " ms" << std::endl;

        // declare a host image object for an 8-bit grayscale image
        npp::ImageCPU_8u_C1 oHostDst(width, height);		

#if (NPP_CUDA_VERSION >= 0x4010)
		// scale the result to map 1 -> 255
		nppiMulC_8u_C1IRSfs(255, oDeviceDst.data(), oDeviceDst.pitch(), size, 0);
#endif

		// and copy the labeling result
		oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

		int E_d, E_s;
		std::cout << "Graphcut Cost: " << computeEnergy(E_d, E_s, oHostDst.data(), oHostDst.pitch(), hCue, vCue, dataCostArray, width, height ) << std::endl;
		std::cout << "(E_d = " << E_d << ", E_s = " << E_s << ")" << std::endl; 

		std::cout << "Saving segmtation result as " << sResultFilename << std::endl;
		saveImage(sResultFilename, oHostDst);

#if (NPP_CUDA_VERSION >= 0x4010)
		nppiGraphcutFree(pGraphcutState);
#endif
		cudaFree(pBuffer);
		cudaFree(d_top);
		cudaFree(d_bottom);
		cudaFree(d_left_transposed);
		cudaFree(d_right_transposed);
		cudaFree(d_source);
		cudaFree(d_sink);
		cudaFree(d_terminals);

        shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
	}
    catch (npp::Exception & rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;
        shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;
        shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
    }
    
    return 0;
}
