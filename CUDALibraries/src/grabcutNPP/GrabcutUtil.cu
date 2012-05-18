/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "nppi.h"

__global__
void TrimapFromRectKernel(Npp8u* alpha, int alpha_pitch, NppiRect rect, int width, int height ) {

	Npp32u* alpha4 = (Npp32u*)alpha;
	int alpha4_pitch = alpha_pitch / 4;

	int x0 = blockIdx.x * 32 + threadIdx.x;
	int y0 = blockIdx.y * 32;

	int x = x0 * 4;

	for( int i=threadIdx.y; i<32; i+=blockDim.y) {

		int y = y0 + i;
		if( x< width && y < height ) {

			if( y >= rect.y && y < (rect.y + rect.height)) {
				int first_x = min(max(0, rect.x - x),4);
				int last_x = min(max(0,x - (rect.x + rect.width - 4)),4);

				unsigned int pattern = 0x001010101u;
				unsigned int mask    = 0x0ffffffffu;

				alpha4[y * alpha4_pitch + x0] = (pattern << (first_x*8)) & (mask >> (last_x*8));
			} else {
				alpha4[y * alpha4_pitch + x0] = 0;
			}

		}
	}

}

cudaError_t TrimapFromRect(Npp8u* alpha, int alpha_pitch, NppiRect rect, int width, int height )
{
	dim3 block(32,8);
	dim3 grid( (width+(block.x*4)-1) / (block.x*4), (height+31) / 32);

	rect.y = height - 1 - (rect.y + rect.height - 1) ; // Flip horizontal (FreeImage inverts y axis)

	TrimapFromRectKernel<<<grid, block>>>(alpha, alpha_pitch, rect, width, height );

	return cudaGetLastError();
}

__device__
uchar4 filter(uchar4 color)
{
	float hue = 0.33f * (color.x + color.y + color.z);

	return make_uchar4(rintf(hue * 0.6f), rintf(hue * 0.3f), rintf(hue * 1.0f), color.w);
}


template<int mode>
__global__
void ApplyMatteKernel(uchar4* result, int result_pitch, const uchar4* image, int image_pitch, const unsigned char* matte, int matte_pitch, int width, int height) {
	
	int x = blockIdx.x * 32 + threadIdx.x;
	int y = blockIdx.y * 32 + threadIdx.y;

	for( int k=0; k < 4; ++k ) {		
		if( x < width && y < height ) {
			uchar4 pixel = image[ y * image_pitch + x];
			
			if( mode == 1 ) {
				if( ((matte[y * matte_pitch + x]) & 1) == 0 ) {
					pixel = filter(pixel);
				}
			}

			if( mode == 2 ) {
				if( ((matte[y * matte_pitch + x]) & 1)  == 0 ) {
					pixel = make_uchar4(0,0,0,0);
				}
			}

			result[y * result_pitch + x] = pixel;
		}
		y += blockDim.y;
	}

}


cudaError_t ApplyMatte(int mode, uchar4* result, int result_pitch, const uchar4* image, int image_pitch, const unsigned char* matte, int matte_pitch, int width, int height)
{
	dim3 block(32,8);
	dim3 grid( (width+31) / 32, (height+31) / 32);
	
	switch(mode) {
		case 0 :
			ApplyMatteKernel<0><<<grid, block>>>(result, result_pitch/4, image, image_pitch/4, matte, matte_pitch, width, height);
			break;

		case 1 :
			ApplyMatteKernel<1><<<grid, block>>>(result, result_pitch/4, image, image_pitch/4, matte, matte_pitch, width, height);
			break;

		case 2 :
			ApplyMatteKernel<2><<<grid, block>>>(result, result_pitch/4, image, image_pitch/4, matte, matte_pitch, width, height);
			break;

	}

	return cudaGetLastError();
}

__global__
void convertRGBToRGBAKernel(uchar4* i4, int i4_pitch, uchar3* i3, int i3_pitch, int width, int height)
{
	int x = blockIdx.x * 32 + threadIdx.x;
	int y = blockIdx.y * 32 + threadIdx.y;

	unsigned char* i3_linear = (unsigned char*) i3;
	unsigned char* i4_linear = (unsigned char*) i4;

	for( int k=0; k < 4;++k ) {
		if( x < width && y < height ) {			
			uchar3* i3_line = (uchar3*)(i3_linear + y*i3_pitch);
			uchar4* i4_line = (uchar4*)(i4_linear + y*i4_pitch);
			
			uchar3 pixel = i3_line[x];
			i4_line[x] = make_uchar4(pixel.x, pixel.y, pixel.z, 255);
		}
		y += blockDim.y;
	}
}


cudaError_t convertRGBToRGBA(uchar4* i4, int i4_pitch, uchar3* i3, int i3_pitch, int width, int height)
{
	dim3 block(32,8);
	dim3 grid((width + 31)/32, (height+31)/32);

	convertRGBToRGBAKernel<<<grid, block>>>(i4, i4_pitch, i3, i3_pitch, width, height);

	return cudaGetLastError();
}

