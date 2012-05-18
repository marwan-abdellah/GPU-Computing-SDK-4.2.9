#include "Signal.h"

namespace npp 
{
    Signal::Signal(): nSize_(0)
    { ; }

    Signal::Signal(size_t nSize): nSize_(nSize)
    { ; }

    Signal::Signal(const Signal & rSignal): nSize_(rSignal.nSize_)
    { ; }

    Signal::~Signal()
    { ; }

    Signal &
    Signal::operator= (const Signal & rSignal)
    {
        nSize_ = rSignal.nSize_;
        
        return *this;
    }

    void
    Signal::swap(Signal & rSignal)
    {
        size_t nTemp = nSize_;
        nSize_ = rSignal.nSize_;
        rSignal.nSize_ = nTemp;
    }
    
}; // npp namespace
