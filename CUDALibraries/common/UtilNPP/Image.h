#ifndef NV_UTIL_NPP_IMAGE_H
#define NV_UTIL_NPP_IMAGE_H
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

#include <cstddef>

namespace npp
{

    class Image
    {
    public:
        struct Size
        {
            unsigned int nWidth;
            unsigned int nHeight;
            
            Size();
            
            Size(unsigned int nWidthNew, unsigned nHeightNew);
            
            Size(const Size & rSize);
            
            Size &
            operator= (const Size & rSize);
            
            void
            swap(Size & rSize);
        };
        
        Image();
        
        Image(unsigned int nWidth, unsigned int nHeight);
        
        Image(const Image::Size & rSize);
        
        Image(const Image & rImage);
        
        virtual
       ~Image();
       
        Image &
        operator= (const Image & rImage);
        
        unsigned int
        width()
        const
        {
            return oSize_.nWidth;
        }
        
        unsigned int
        height()
        const
        {
            return oSize_.nHeight;
        }
        
        Size
        size()
        const
        {
            return oSize_;
        }
        
        void
        swap(Image & rImage);

    private:
        Size oSize_;
    };

    bool
    operator== (const Image::Size & rFirst, const Image::Size & rSecond);

    bool
    operator!= (const Image::Size & rFirst, const Image::Size & rSecond);

} // npp namespace 


#endif // NV_UTIL_NPP_IMAGE_H
