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

#ifdef WIN32
#include <windows.h>
#endif

#include <npp.h>

#include "FreeImage.h"
#include "Exceptions.h"

#include <string.h>
#include <fstream>
#include <iostream>

#include <shrQATest.h>

#ifdef WIN32
#define STRCASECMP  _stricmp
#define STRNCASECMP _strnicmp
#else
#define STRCASECMP  strcasecmp
#define STRNCASECMP strncasecmp
#endif

bool g_bQATest = false;
int  g_nDevice = -1;

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

inline void cudaSafeCallNoSync( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
        std::cerr << file << "(" << line << ")" << " : cudaSafeCallNoSync() Runtime API error : " << cudaGetErrorString(err) << std::endl;
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
        printf("%s using GPU <%s> with %d SM(s) with", argv[0], nppGetGpuName(), nppGetGpuNumSMs(), sComputeCap[computeCap]);
	if (computeCap > 0) {
            printf(" %s\n", sComputeCap[computeCap]);
	} else {
	    printf(" Unknwon Compute Capabilities\n");
        }	
    } else {
        printf("%s\n", sComputeCap[computeCap]);
    }
}

// Error handler for FreeImage library. 
//  In case this handler is invoked, it throws an NPP exception.
void 
FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char * zMessage) 
{
    throw npp::Exception(zMessage);
}

std::ostream &
operator <<(std::ostream & rOutputStream, const FIBITMAP & rBitmap)
{
    unsigned int nImageWidth    = FreeImage_GetWidth(const_cast<FIBITMAP *>(&rBitmap));
    unsigned int nImageHeight   = FreeImage_GetHeight(const_cast<FIBITMAP *>(&rBitmap));
    unsigned int nPitch         = FreeImage_GetPitch(const_cast<FIBITMAP *>(&rBitmap));
    unsigned int nBPP           = FreeImage_GetBPP(const_cast<FIBITMAP *>(&rBitmap));
    
    FREE_IMAGE_COLOR_TYPE eType = FreeImage_GetColorType(const_cast<FIBITMAP *>(&rBitmap));
    BITMAPINFO * pInfo          = FreeImage_GetInfo(const_cast<FIBITMAP *>(&rBitmap));
    
    rOutputStream << "Size  (" << FreeImage_GetWidth(const_cast<FIBITMAP *>(&rBitmap)) << ", " 
                               << FreeImage_GetHeight(const_cast<FIBITMAP *>(&rBitmap)) << ")\n";
    rOutputStream << "Pitch "  << FreeImage_GetPitch(const_cast<FIBITMAP *>(&rBitmap)) << "\n";
    rOutputStream << "Type  ";
    switch (eType)
    {
    case FIC_MINISWHITE:
        rOutputStream << "FIC_MINISWHITE\n";
    break;
    case FIC_MINISBLACK:
        rOutputStream << "FIC_MINISBLACK\n";
    break;
    case FIC_RGB:
        rOutputStream << "FIC_RGB\n";
    break;
    case FIC_PALETTE:
        rOutputStream << "FIC_PALETTE\n";
    break;
    case FIC_RGBALPHA:
        rOutputStream << "FIC_RGBALPHA\n";
    break;
    case FIC_CMYK:
        rOutputStream << "FIC_CMYK\n";
    break;
    default:
        rOutputStream << "Unknown pixel format.\n";
    }
    rOutputStream << "BPP   " << nBPP << std::endl;
    
    return rOutputStream;
}

int 
main(int argc, char *argv[]) 
{
    shrQAStart(argc, argv);

    try
    {
        std::string sFilename;
        char *filePath = findFilePath("Lena.pgm", argv[0]);
        if (filePath) {
             sFilename = filePath;
        } else {
             printf("Error unable to find Lena.pgm\n");
             shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
	}

        // set your own FreeImage error handler
        FreeImage_SetOutputMessage(FreeImageErrorHandler);

        // Parse the command line arguments for proper configuration
        parseCommandLineArguments(argc, argv);

        printfNPPinfo(argc, argv);

        if (g_bQATest == false && (g_nDevice == -1) && argc > 1) {
            sFilename = argv[1];
        }

        // if we specify the filename at the command line, then we only test sFilename
        // otherwise we will check both sFilename[0,1] 
        int file_errors = 0;
        std::ifstream infile(sFilename.data(), std::ifstream::in);
        if (infile.good()) {
            std::cout << "freeImageInteropNPP opened: <" << sFilename.data() << "> successfully!" << std::endl;
            file_errors = 0;
            infile.close();
        } else {
            std::cout << "freeImageInteropNPP unable to open: <" << sFilename.data() << ">" << std::endl;
            file_errors++;
            infile.close();
        }
        if (file_errors > 0) {
            shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
        }

        std::string sResultFilename = sFilename;
        
        std::string::size_type dot = sResultFilename.rfind('.');
        if (dot != std::string::npos) sResultFilename = sResultFilename.substr(0, dot);
        sResultFilename += "_boxFilterFII.pgm";

        if (argc >= 3 && !g_bQATest)
            sResultFilename = argv[2];

        FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(sFilename.c_str());
                // no signature? try to guess the file format from the file extension
        if (eFormat == FIF_UNKNOWN)
            eFormat = FreeImage_GetFIFFromFilename(sFilename.c_str());
        NPP_ASSERT(eFormat != FIF_UNKNOWN);
                // check that the plugin has reading capabilities ...
        FIBITMAP * pBitmap;
        if (FreeImage_FIFSupportsReading(eFormat)) 
            pBitmap = FreeImage_Load(eFormat, sFilename.c_str());
        NPP_ASSERT(pBitmap != 0);
                // Dump the bitmap information to the console
        std::cout << (*pBitmap) << std::endl;
                // make sure this is an 8-bit single channel image
        NPP_ASSERT(FreeImage_GetColorType(pBitmap) == FIC_MINISBLACK);
        NPP_ASSERT(FreeImage_GetBPP(pBitmap) == 8);
        
        unsigned int nImageWidth  = FreeImage_GetWidth(pBitmap);
        unsigned int nImageHeight = FreeImage_GetHeight(pBitmap);
        unsigned int nSrcPitch    = FreeImage_GetPitch(pBitmap);
        unsigned char * pSrcData  = FreeImage_GetBits(pBitmap);
        
        int nSrcPitchCUDA;
        Npp8u * pSrcImageCUDA = nppiMalloc_8u_C1(nImageWidth, nImageHeight, &nSrcPitchCUDA);
        NPP_ASSERT_NOT_NULL(pSrcImageCUDA);
                // copy image loaded via FreeImage to into CUDA device memory, i.e.
                // transfer the image-data up to the GPU's video-memory
        NPP_CHECK_CUDA(cudaMemcpy2D(pSrcImageCUDA, nSrcPitchCUDA, pSrcData, nSrcPitch, 
                                nImageWidth, nImageHeight, cudaMemcpyHostToDevice));
                // define size of the box filter
        const NppiSize  oMaskSize   = {7, 7};
        const NppiPoint oMaskAchnor = {0, 0};
                // compute maximal result image size
        const NppiSize  oSizeROI = {nImageWidth  - (oMaskSize.width - 1),
                                     nImageHeight - (oMaskSize.height - 1)};
                // allocate result image memory
        int nDstPitchCUDA; 
        Npp8u * pDstImageCUDA = nppiMalloc_8u_C1(oSizeROI.width, oSizeROI.height, &nDstPitchCUDA);
        NPP_ASSERT_NOT_NULL(pDstImageCUDA);
        NPP_CHECK_NPP(nppiFilterBox_8u_C1R(pSrcImageCUDA, nSrcPitchCUDA, pDstImageCUDA, nDstPitchCUDA, 
                                         oSizeROI, oMaskSize, oMaskAchnor));
                // create the result image storage using FreeImage so we can easily 
                // save
        FIBITMAP * pResultBitmap = FreeImage_Allocate(oSizeROI.width, oSizeROI.height, 8 /* bits per pixel */);
        NPP_ASSERT_NOT_NULL(pResultBitmap);
        unsigned int nResultPitch   = FreeImage_GetPitch(pResultBitmap);
        unsigned char * pResultData = FreeImage_GetBits(pResultBitmap);

        NPP_CHECK_CUDA(cudaMemcpy2D(pResultData, nResultPitch, pDstImageCUDA, nDstPitchCUDA,
                                oSizeROI.width, oSizeROI.height, cudaMemcpyDeviceToHost));
                // now save the result image
        bool bSuccess;
        bSuccess = FreeImage_Save(FIF_PGM, pResultBitmap, sResultFilename.c_str(), 0) == TRUE;
        NPP_ASSERT_MSG(bSuccess, "Failed to save result image.");
		
        cudaDeviceReset();
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
    
	shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
}
