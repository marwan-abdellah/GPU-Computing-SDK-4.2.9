#ifndef NV_UTIL_NPP_IMAGES_NPP_H
#define NV_UTIL_NPP_IMAGES_NPP_H
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

#include "Exceptions.h"
#include "ImagePacked.h"

#include "ImageAllocatorsNPP.h"
#include <cuda_runtime.h>

namespace npp
{
            // forward declaration
    template<typename D, unsigned int N, class A> class ImageCPU;

    template<typename D, unsigned int N>
    class ImageNPP: public npp::ImagePacked<D, N, npp::ImageAllocator<D, N> >
    {
    public:
        ImageNPP()
        { ; }
        
        ImageNPP(unsigned int nWidth, unsigned int nHeight, bool bTight = false): ImagePacked<D, N, npp::ImageAllocator<D, N> >(nWidth, nHeight, bTight)
        { ; }

        ImageNPP(const npp::Image::Size & rSize): ImagePacked<D, N, npp::ImageAllocator<D, N> >(rSize)
        { ; }

        ImageNPP(const ImageNPP<D, N> & rImage): Image(rImage)
        { ; }
        
        template<class X>
        explicit
        ImageNPP(const ImageCPU<D, N, X> & rImage, bool bTight = false): ImagePacked<D, N, npp::ImageAllocator<D, N> >(rImage.width(), rImage.height(), bTight)
        {
            npp::ImageAllocator<D, N>::HostToDeviceCopy2D(ImagePacked<D, N, npp::ImageAllocator<D, N> >::data(), 
                                                          ImagePacked<D, N, npp::ImageAllocator<D, N> >::pitch(), 
                                                          rImage.data(), 
                                                          rImage.pitch(), 
                                                          ImagePacked<D, N, npp::ImageAllocator<D, N> >::width(), 
                                                          ImagePacked<D, N, npp::ImageAllocator<D, N> >::height());
        }

        virtual
       ~ImageNPP()
        { ; }
        
        ImageNPP &
        operator= (const ImageNPP<D, N> & rImage)
        {
            ImagePacked<D, N, npp::ImageAllocator<D, N> >::operator= (rImage);
            
            return *this;
        }
        
        void
        copyTo(D * pData, unsigned int nPitch)
        const
        {
            NPP_ASSERT((ImagePacked<D, N, npp::ImageAllocator<D, N> >::width() * sizeof(npp::Pixel<D, N>) <= nPitch));
            npp::ImageAllocator<D, N>::DeviceToHostCopy2D(pData, 
                                                          nPitch, 
                                                          ImagePacked<D, N, npp::ImageAllocator<D, N> >::data(), 
                                                          ImagePacked<D, N, npp::ImageAllocator<D, N> >::pitch(), 
                                                          ImagePacked<D, N, npp::ImageAllocator<D, N> >::width(), 
                                                          ImagePacked<D, N, npp::ImageAllocator<D, N> >::height());
        }

	    void
	    copyFrom(D * pData, unsigned int nPitch) 
	    {
		    NPP_ASSERT((ImagePacked<D, N, npp::ImageAllocator<D, N> >::width() * sizeof(npp::Pixel<D, N>) <= nPitch));
		    npp::ImageAllocator<D, N>::HostToDeviceCopy2D(ImagePacked<D, N, npp::ImageAllocator<D, N> >::data(), 
		                                                  ImagePacked<D, N, npp::ImageAllocator<D, N> >::pitch(), 
		                                                  pData, 
		                                                  nPitch, 
		                                                  ImagePacked<D, N, npp::ImageAllocator<D, N> >::width(), 
		                                                  ImagePacked<D, N, npp::ImageAllocator<D, N> >::height());
	    }
    };

    typedef ImageNPP<Npp8u,  1>   ImageNPP_8u_C1;
    typedef ImageNPP<Npp8u,  2>   ImageNPP_8u_C2;
    typedef ImageNPP<Npp8u,  3>   ImageNPP_8u_C3;
    typedef ImageNPP<Npp8u,  4>   ImageNPP_8u_C4;

    typedef ImageNPP<Npp16u, 1>  ImageNPP_16u_C1;
    typedef ImageNPP<Npp16u, 2>  ImageNPP_16u_C2;
    typedef ImageNPP<Npp16u, 3>  ImageNPP_16u_C3;
    typedef ImageNPP<Npp16u, 4>  ImageNPP_16u_C4;

    typedef ImageNPP<Npp16s, 1>  ImageNPP_16s_C1;
    typedef ImageNPP<Npp16s, 3>  ImageNPP_16s_C3;
    typedef ImageNPP<Npp16s, 4>  ImageNPP_16s_C4;

    typedef ImageNPP<Npp32s, 1>  ImageNPP_32s_C1;
    typedef ImageNPP<Npp32s, 3>  ImageNPP_32s_C3;
    typedef ImageNPP<Npp32s, 4>  ImageNPP_32s_C4;

    typedef ImageNPP<Npp32f, 1>  ImageNPP_32f_C1;
    typedef ImageNPP<Npp32f, 2>  ImageNPP_32f_C2;
    typedef ImageNPP<Npp32f, 3>  ImageNPP_32f_C3;
    typedef ImageNPP<Npp32f, 4>  ImageNPP_32f_C4;
    
    typedef ImageNPP<Npp64f, 1>  ImageNPP_64f_C1;
    typedef ImageNPP<Npp64f, 2>  ImageNPP_64f_C2;
    typedef ImageNPP<Npp64f, 3>  ImageNPP_64f_C3;
    typedef ImageNPP<Npp64f, 4>  ImageNPP_64f_C4;

} // npp namespace

#endif // NV_UTIL_NPP_IMAGES_NPP_H
