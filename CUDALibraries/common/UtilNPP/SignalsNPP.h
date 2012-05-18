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
#ifndef NV_UTIL_NPP_SIGNALS_NPP_H
#define NV_UTIL_NPP_SIGNALS_NPP_H

#include "Exceptions.h"
#include "Signal.h"

#include "SignalAllocatorsNPP.h"
#include <cuda_runtime.h>

namespace npp
{
            // forward declaration
    template<typename D, class A> class SignalCPU;

    template<typename D>
    class SignalNPP: public npp::SignalTemplate<D, npp::SignalAllocator<D> >
    {
    public:
        SignalNPP()
        { ; }
        
        explicit
        SignalNPP(size_t nSize): SignalTemplate<D, npp::SignalAllocator<D> >(nSize)
        { ; }

        SignalNPP(const SignalNPP<D> & rSignal): SignalTemplate<D, npp::SignalAllocator<D> >(rSignal)
        { ; }
        
        template<class X>
        explicit
        SignalNPP(const SignalCPU<D, X> & rSignal): SignalTemplate<D, npp::SignalAllocator<D> >(rSignal.size())
        {
            npp::SignalAllocator<D>::HostToDeviceCopy1D(SignalTemplate<D, npp::SignalAllocator<D> >::values(), 
                                                        rSignal.values(), SignalTemplate<D, npp::SignalAllocator<D> >::size());
        }

        virtual
       ~SignalNPP()
        { ; }
        
        SignalNPP &
        operator= (const SignalNPP<D> & rSignal)
        {
            SignalTemplate<D, npp::SignalAllocator<D> >::operator= (rSignal);
            
            return *this;
        }
        
        void
        copyTo(D * pValues)
        const
        {
            npp::SignalAllocator<D>::DeviceToHostCopy1D(pValues, SignalTemplate<D, npp::SignalAllocator<D> >::values(), SignalTemplate<D, npp::SignalAllocator<D> >::size());
        }

	    void
	    copyFrom(D * pValues) 
	    {
		    npp::SignalAllocator<D>::HostToDeviceCopy1D(SignalTemplate<D, npp::SignalAllocator<D> >::values(), pValues, SignalTemplate<D, npp::SignalAllocator<D> >::size());
	    }
    };

    typedef SignalNPP<Npp8u>    SignalNPP_8u;
    typedef SignalNPP<Npp16s>   SignalNPP_16s;
    typedef SignalNPP<Npp16sc>  SignalNPP_16sc;
    typedef SignalNPP<Npp32s>   SignalNPP_32s;
    typedef SignalNPP<Npp32sc>  SignalNPP_32sc;
    typedef SignalNPP<Npp32f>   SignalNPP_32f;
    typedef SignalNPP<Npp32fc>  SignalNPP_32fc;  
    typedef SignalNPP<Npp64s>   SignalNPP_64s;
    typedef SignalNPP<Npp64sc>  SignalNPP_64sc; 
    typedef SignalNPP<Npp64f>   SignalNPP_64f;   
    typedef SignalNPP<Npp64fc>  SignalNPP_64fc; 
      
} // npp namespace

#endif // NV_UTIL_NPP_SIGNALS_NPP_H
