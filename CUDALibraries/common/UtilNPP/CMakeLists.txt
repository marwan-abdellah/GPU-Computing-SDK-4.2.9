CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

include_directories(${CUDA_INCLUDE_DIRS} . ../../../npp/include/ ../FreeImage/include)

SET(UtilNPP_HEADERS
    Exceptions.h
    Image.h
    ImageAllocatorsCPU.h
    ImageAllocatorsNPP.h
    ImageIO.h
    ImagePacked.h
    ImagesCPU.h
    ImagesNPP.h
    Pixel.h
    Signal.h
    SignalAllocatorsCPU.h
    SignalAllocatorsNPP.h
    SignalsCPU.h
    SignalsNPP.h
    StopWatch.h
    StopWatchBase.h
    StopWatchBase.inl
    StopWatchLinux.h
    StopWatchWin.h   
)


SET(UtilNPP_SOURCES
    Exceptions.cpp
    Image.cpp
    ImageIO.cpp
    Signal.cpp
)

IF(WIN32)
    SET(StopWatch_SOURCES StopWatchWin.cpp )
ELSE(WIN32)
    SET(StopWatch_SOURCES StopWatchLinux.cpp )
ENDIF(WIN32)

#source_group("Header Files" FILES ${UtilNPP_HEADERS})

ADD_LIBRARY(UtilNPP STATIC ${UtilNPP_HEADERS} ${UtilNPP_SOURCES} ${StopWatch_SOURCES})
