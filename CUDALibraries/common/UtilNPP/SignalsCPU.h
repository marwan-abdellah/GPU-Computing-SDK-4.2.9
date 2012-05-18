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
#ifndef NV_UTIL_NPP_SIGNALS_CPU_H
#define NV_UTIL_NPP_SIGNALS_CPU_H

#include "Signal.h"

#include "SignalAllocatorsCPU.h"
#include "Exceptions.h"

#include <npp.h>


namespace npp
{

    template<typename D, class A>
    class SignalCPU: public npp::SignalTemplate<D, A>
    {
    public:
        typedef typename npp::SignalTemplate<D, A>::tData tData;
        
        SignalCPU()
        { ; }
        
        SignalCPU(size_t nSize): SignalTemplate<D, A>(nSize)
        { ; }
        
        SignalCPU(const SignalCPU<D, A> & rSignal): SignalTemplate<D, A>(rSignal)
        { ; }
        
        virtual
       ~SignalCPU()
        { ; }

        SignalCPU &
        operator= (const SignalCPU<D,A> & rSignal)
        {
            SignalTemplate<D, A>::operator= (rSignal);
            
            return *this;
        }
        
        tData &
        operator [](unsigned int i)
        {
            return *SignalTemplate<D, A>::values(i);
        }
        
        tData
        operator [](unsigned int i)
        const
        {
            return *SignalTemplate<D, A>::values(i);
        }
        
    };

    typedef SignalCPU<Npp8u,   npp::SignalAllocatorCPU<Npp8u>   >   SignalCPU_8u;
    typedef SignalCPU<Npp32s,  npp::SignalAllocatorCPU<Npp32s>  >   SignalCPU_32s;
    typedef SignalCPU<Npp16s,  npp::SignalAllocatorCPU<Npp16s>  >   SignalCPU_16s;
    typedef SignalCPU<Npp16sc, npp::SignalAllocatorCPU<Npp16sc> >   SignalCPU_16sc;
    typedef SignalCPU<Npp32sc, npp::SignalAllocatorCPU<Npp32sc> >   SignalCPU_32sc;
    typedef SignalCPU<Npp32f,  npp::SignalAllocatorCPU<Npp32f>  >   SignalCPU_32f;
    typedef SignalCPU<Npp32fc, npp::SignalAllocatorCPU<Npp32fc> >   SignalCPU_32fc;  
    typedef SignalCPU<Npp64s,  npp::SignalAllocatorCPU<Npp64s>  >   SignalCPU_64s;
    typedef SignalCPU<Npp64sc, npp::SignalAllocatorCPU<Npp64sc> >   SignalCPU_64sc;  
    typedef SignalCPU<Npp64f,  npp::SignalAllocatorCPU<Npp64f>  >   SignalCPU_64f;  
    typedef SignalCPU<Npp64fc, npp::SignalAllocatorCPU<Npp64fc> >   SignalCPU_64fc;  
    
} // npp namespace

#endif // NV_UTIL_NPP_SIGNALS_CPU_H
