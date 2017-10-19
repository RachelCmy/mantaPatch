/******************************************************************************
*
* MantaFlow fluid solver framework
* Copyright 2017 Steffen Wiewel, Moritz Baecher, Rachel Chu
*
* This program is free software, distributed under the terms of the
* GNU General Public License (GPL) 
* http://www.gnu.org/licenses
*
* Convert mantaflow grids to/from numpy arrays
*
******************************************************************************/

#ifdef _PCONVERT_H
#ifndef _NUMPYCONVERT_H
#define _NUMPYCONVERT_H

enum NumpyTypes
{
	N_BOOL = 0,
	N_BYTE, N_UBYTE,
	N_SHORT, N_USHORT,
	N_INT, N_UINT,
	N_LONG, N_ULONG,
	N_LONGLONG, N_ULONGLONG,
	N_FLOAT, N_DOUBLE, N_LONGDOUBLE,
	N_CFLOAT, N_CDOUBLE, N_CLONGDOUBLE,
	N_OBJECT = 17,
	N_STRING, N_UNICODE,
	N_VOID,
	/*
	* New 1.6 types appended, may be integrated
	* into the above in 2.0.
	*/
	N_DATETIME, N_TIMEDELTA, N_HALF,

	N_NTYPES,
	N_NOTYPE,
	N_CHAR,      /* special flag */
	N_USERDEF = 256,  /* leave room for characters */

	/* The number of types not including the new 1.6 types */
	N_NTYPES_ABI_COMPATIBLE = 21
};

namespace Manta
{
    struct PyArrayContainer
    {
        unsigned int TotalSize;
        NumpyTypes DataType;
        void* pData;

		// should only be called when the data type matches!
		// Real - N_FLOAT : single precision
		// Real - N_DOUBLE : double precision
		// int  - N_INT : 32/64
		// Vec3 - same as Real
		//! access data
		void get(IndexInt idx, Real* &data);
		void get(IndexInt idx, int* &data);
		void get(IndexInt idx, Vec3* &data);
    };
    
    template<> PyArrayContainer* fromPyPtr<PyArrayContainer>(PyObject* obj, std::vector<void*>* tmp);
    template<> PyArrayContainer fromPy<PyArrayContainer>(PyObject* obj);
} // namespace

#endif
#endif
