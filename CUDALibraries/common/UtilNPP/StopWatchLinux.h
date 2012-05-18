#ifndef NV_NPP_UTIL_STOP_WATCH_LINUX_H
#define NV_NPP_UTIL_STOP_WATCH_LINUX_H
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

// includes, system
#include <ctime>
#include <sys/time.h>

namespace npp
{

    /// Windows specific implementation of StopWatch
    class StopWatchLinux {

    protected:

        //! Constructor, default
        StopWatchLinux();

        // Destructor
        ~StopWatchLinux();

    public:

        //! Start time measurement
        inline void start();

        //! Stop time measurement
        inline void stop();

        //! Reset time counters to zero
        inline void reset();

        //! Time in msec. after start. If the stop watch is still running (i.e. there
        //! was no call to stop()) then the elapsed time is returned, otherwise the
        //! time between the last start() and stop call is returned
        inline const double elapsed() const;


    private:

        // helper functions
      
        //! Get difference between start time and current time
        inline double getDiffTime() const;

    private:

        // member variables

        //! Start of measurement
        struct timeval  start_time;

        //! Time difference between the last start and stop
        double  diff_time;

        //! TOTAL time difference between starts and stops
        double  total_time;

        //! flag if the stop watch is running
        bool running;
    };

    // functions, inlined

    ////////////////////////////////////////////////////////////////////////////////
    //! Start time measurement
    ////////////////////////////////////////////////////////////////////////////////
    inline void
    StopWatchLinux::start() {

      gettimeofday( &start_time, 0);
      running = true;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Stop time measurement and increment add to the current diff_time summation
    //! variable. Also increment the number of times this clock has been run.
    ////////////////////////////////////////////////////////////////////////////////
    inline void
    StopWatchLinux::stop() {

      diff_time = getDiffTime();
      total_time += diff_time;
      running = false;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Reset the timer to 0. Does not change the timer running state but does 
    //! recapture this point in time as the current start time if it is running.
    ////////////////////////////////////////////////////////////////////////////////
    inline void
    StopWatchLinux::reset() 
    {
      diff_time = 0;
      total_time = 0;
      if( running )
        gettimeofday( &start_time, 0);
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Time in msec. after start. If the stop watch is still running (i.e. there
    //! was no call to stop()) then the elapsed time is returned added to the 
    //! current diff_time sum, otherwise the current summed time difference alone
    //! is returned.
    ////////////////////////////////////////////////////////////////////////////////
    inline const double 
    StopWatchLinux::elapsed() const 
    {
        // Return the TOTAL time to date
        double retval = total_time;
        if( running) {

            retval += getDiffTime();
        }

        return retval;
    }



    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    inline double
    StopWatchLinux::getDiffTime() const 
    {
      struct timeval t_time;
      gettimeofday( &t_time, 0);

      // time difference in milli-seconds
      return   (1000.0 * ( t_time.tv_sec - start_time.tv_sec) 
                + (0.001 * (t_time.tv_usec - start_time.tv_usec)) );
    }

} // npp namespace 

#endif // NV_NPP_UTIL_STOP_WATCH_LINUX_H

