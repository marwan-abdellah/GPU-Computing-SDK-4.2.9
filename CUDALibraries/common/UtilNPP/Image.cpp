#include "Image.h"

#include "Exceptions.h"

namespace npp 
{

    Image::Size::Size(): nWidth(0)
                       , nHeight(0)
    { ; }
            
    Image::Size::Size(unsigned int nWidthNew, unsigned nHeightNew): nWidth(nWidthNew)
                                                                  , nHeight(nHeightNew)
    { ; }

    Image::Size::Size(const Size & rSize): nWidth(rSize.nWidth)
                                         , nHeight(rSize.nHeight)
    { ; }
            
    Image::Size &
    Image::Size::operator= (const Size & rSize)
    {
        if (&rSize == this)
            return *this;
        
        nWidth = rSize.nWidth;
        nHeight = rSize.nHeight;
        
        return *this;
    }
            
    void
    Image::Size::swap(Size & rSize)
    {
        unsigned int nTemp;
        nTemp = nWidth;
        nWidth = rSize.nWidth;
        rSize.nWidth = nTemp;
        
        nTemp = nHeight;
        nHeight = rSize.nHeight;
        rSize.nHeight = nTemp;
    }


    Image::Image()
    { ; }

    Image::Image(unsigned int nWidth, unsigned int nHeight): oSize_(nWidth, nHeight)
    { ; }

    Image::Image(const Image::Size & rSize): oSize_(rSize)
    { ; }

    Image::Image(const Image & rImage): oSize_(rImage.oSize_)
    { ; }
        

    Image::~Image()
    { ; }

    Image &
    Image::operator= (const Image & rImage)
    {
        if (&rImage == this)
            return *this;
        
        oSize_  = rImage.oSize_;
        
        return *this;
    }
        
    void
    Image::swap(Image & rImage)
    {
        oSize_.swap(rImage.oSize_);
    }

    bool
    operator== (const Image::Size & rFirst, const Image::Size & rSecond)
    {
        return rFirst.nWidth == rSecond.nWidth && rFirst.nHeight == rSecond.nHeight;
    }

    bool
    operator!= (const Image::Size & rFirst, const Image::Size & rSecond)
    {
        return rFirst.nWidth != rSecond.nWidth || rFirst.nHeight != rSecond.nHeight;
    }


} // npp namespace 
