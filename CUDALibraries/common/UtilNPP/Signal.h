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
#ifndef NV_UTIL_NPP_SIGNAL_H
#define NV_UTIL_NPP_SIGNAL_H

#include <cstring>

namespace npp
{
    class Signal
    {
    public:
        Signal();
        
        explicit
        Signal(size_t nSize);
        
        Signal(const Signal & rSignal);
        
        virtual
       ~Signal();
       
        Signal &
        operator= (const Signal & rSignal);
        
        size_t
        size()
        const
        {
            return nSize_;
        }
        
        void
        swap(Signal & rSignal);

    private:
        size_t nSize_;
    };

    template<typename D, class A>
    class SignalTemplate: public Signal
    {
    public:
        typedef D tData;

        SignalTemplate(): aValues_(0)
        { ; }
        
        SignalTemplate(size_t nSize): Signal(nSize)
                                    , aValues_(0)
        {
            aValues_ = A::Malloc1D(size());
        }
        
        SignalTemplate(const SignalTemplate<D, A> & rSignal): Signal(rSignal)
                                                            , aValues_(0)
        {
            aValues_ = A::Malloc1D(size());
            A::Copy1D(aValues_, rSignal.values(), size());
        }
        
        virtual
       ~SignalTemplate()
        {
            A::Free1D(aValues_);
        }

        SignalTemplate &
        operator= (const SignalTemplate<D, A> & rSignal)
        {
                    // in case of self-assignment
            if (&rSignal == this)
                return *this;
                
            A::Free1D(aValues_);
            this->aPixels_ = 0;
                
                    // assign parent class's data fields (width, height)
            Signal::operator =(rSignal);
                
            aValues_ = A::Malloc1D(size());
            A::Copy1D(aValues_, rSignal.value(), size());
            
            return *this;
        }
        
        /// Get a pointer to the pixel array.
        ///     The result pointer can be offset to pixel at position (x, y) and 
        /// even negative offsets are allowed.
        /// \param nX Horizontal pointer/array offset.
        /// \param nY Vertical pointer/array offset.
        /// \return Pointer to the pixel array (or first pixel in array with coordinates (nX, nY).
        tData * 
        values(int i = 0)
        {
            return aValues_ + i;
        }
        
        const 
        tData * 
        values(int i = 0)
        const
        {
            return aValues_ + i;
        }

        void
        swap(SignalTemplate<D, A> & rSignal)
        {
            Signal::swap(rSignal);
            
            tData * aTemp       = this->aValues_;
            this->aValues_      = rSignal.aValues_;
            rSignal.aValues_    = aTemp;
        }

    private:
        D * aValues_;
    };

} // npp namespace 


#endif // NV_UTIL_NPP_SIGNAL_H
