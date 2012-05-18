/*
* Copyright 2008-2009 NVIDIA Corporation.  All rights reserved.
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

#include "ImageIO.h"

#include "FreeImage.h"
#include "Exceptions.h"

#include <string>
#include "string.h"


// Error handler for FreeImage library. 
//  In case this handler is invoked, it throws an NPP exception.
void 
FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char * zMessage) 
{
    throw npp::Exception(zMessage);
}

namespace npp
{

    /// Load a gray-scale image from disk.
    void 
    loadImage(const std::string & rFileName, ImageCPU_8u_C1 & rImage)
    {
	            // set your own FreeImage error handler
	    FreeImage_SetOutputMessage(FreeImageErrorHandler);

        FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(rFileName.c_str());
                // no signature? try to guess the file format from the file extension
        if (eFormat == FIF_UNKNOWN)
            eFormat = FreeImage_GetFIFFromFilename(rFileName.c_str());
        NPP_ASSERT(eFormat != FIF_UNKNOWN);
                // check that the plugin has reading capabilities ...
        FIBITMAP * pBitmap;
        if (FreeImage_FIFSupportsReading(eFormat)) 
            pBitmap = FreeImage_Load(eFormat, rFileName.c_str());
        NPP_ASSERT(pBitmap != 0);
                // make sure this is an 8-bit single channel image
        NPP_ASSERT(FreeImage_GetColorType(pBitmap) == FIC_MINISBLACK);
        NPP_ASSERT(FreeImage_GetBPP(pBitmap) == 8);
        
                // create an ImageCPU to receive the loaded image data
        ImageCPU_8u_C1 oImage(FreeImage_GetWidth(pBitmap), FreeImage_GetHeight(pBitmap));
        
                // Copy the FreeImage data into the new ImageCPU
        unsigned int nSrcPitch = FreeImage_GetPitch(pBitmap);
        const Npp8u * pSrcLine = FreeImage_GetBits(pBitmap) + nSrcPitch * (FreeImage_GetHeight(pBitmap) -1);
        Npp8u * pDstLine = oImage.data();
        unsigned int nDstPitch = oImage.pitch();
        for (size_t iLine = 0; iLine < oImage.height(); ++iLine)
        {
            memcpy(pDstLine, pSrcLine, oImage.width() * sizeof(Npp8u));
            pSrcLine -= nSrcPitch;
            pDstLine += nDstPitch;
        }
        
                // swap the user given image with our result image, effecively
                // moving our newly loaded image data into the user provided shell
        oImage.swap(rImage);
    }
         
    // Save a gray-scale image to disk.
    void 
    saveImage(const std::string & rFileName, const ImageCPU_8u_C1 & rImage)
    {
                // create the result image storage using FreeImage so we can easily 
                // save
        FIBITMAP * pResultBitmap = FreeImage_Allocate(rImage.width(), rImage.height(), 8 /* bits per pixel */);
        NPP_ASSERT_NOT_NULL(pResultBitmap);
        unsigned int nDstPitch   = FreeImage_GetPitch(pResultBitmap);
        Npp8u * pDstLine = FreeImage_GetBits(pResultBitmap) + nDstPitch * (rImage.height()-1);
        const Npp8u * pSrcLine = rImage.data();
        unsigned int nSrcPitch = rImage.pitch();
        
        for (size_t iLine = 0; iLine < rImage.height(); ++iLine)
        {
            memcpy(pDstLine, pSrcLine, rImage.width() * sizeof(Npp8u));
            pSrcLine += nSrcPitch;
            pDstLine -= nDstPitch;
        }
            
                // now save the result image
        bool bSuccess;
        bSuccess = FreeImage_Save(FIF_PGM, pResultBitmap, rFileName.c_str(), 0) == TRUE;
        NPP_ASSERT_MSG(bSuccess, "Failed to save result image.");
    }
    
        /// Load a gray-scale image from disk.
    void 
    loadImage(const std::string & rFileName, ImageNPP_8u_C1 & rImage)
    {
        ImageCPU_8u_C1 oImage;
        loadImage(rFileName, oImage);
        ImageNPP_8u_C1 oResult(oImage);
        rImage.swap(oResult);
    }
         
    // Save an gray-scale image to disk.
    void 
    saveImage(const std::string & rFileName, const ImageNPP_8u_C1 & rImage)
    {
        ImageCPU_8u_C1 oHostImage(rImage.size());
                // copy the device result data
        rImage.copyTo(oHostImage.data(), oHostImage.pitch());
        saveImage(rFileName, oHostImage);
    }

}
