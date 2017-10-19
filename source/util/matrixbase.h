/******************************************************************************
*
* MantaFlow fluid solver framework
* Copyright 2011 Tobias Pfaff, Nils Thuerey
*
* This program is free software, distributed under the terms of the
* GNU General Public License (GPL)
* http://www.gnu.org/licenses
*
* Basic simple matrix class
*
******************************************************************************/


#ifndef _MATRICES_H
#define _MATRICES_H

#include "vectorbase.h"
#include <algorithm>
#include <string.h>

using std::swap;

namespace Manta {


// a basic simple 4x4 matrix class
template<class T>
class Matrix4x4{
	public:
		// Constructor
		inline Matrix4x4(void );
		// Copy-Constructor
		inline Matrix4x4(const Matrix4x4<T> &v );
		// construct a matrix from one T
		inline Matrix4x4(T);
		// construct a matrix from three Vector3D<T>s
		inline Matrix4x4(Vector3D<T>, Vector3D<T>, Vector3D<T>);
		// construct a matrix from scalar pointer 
		inline Matrix4x4(const T*);

		// Assignment operator
		inline const Matrix4x4<T>& operator=  (const Matrix4x4<T>& v);
		// Assignment operator
		inline const Matrix4x4<T>& operator=  (T s);
		// Assign and add operator
		inline const Matrix4x4<T>& operator+= (const Matrix4x4<T>& v);
		// Assign and add operator
		inline const Matrix4x4<T>& operator+= (T s);
		// Assign and sub operator
		inline const Matrix4x4<T>& operator-= (const Matrix4x4<T>& v);
		// Assign and sub operator
		inline const Matrix4x4<T>& operator-= (T s);
		// Assign and mult operator
		inline const Matrix4x4<T>& operator*= (const Matrix4x4<T>& v);
		// Assign and mult operator
		inline const Matrix4x4<T>& operator*= (T s);
		// Assign and div operator
		inline const Matrix4x4<T>& operator/= (const Matrix4x4<T>& v);
		// Assign and div operator
		inline const Matrix4x4<T>& operator/= (T s);


		// unary operator
		inline Matrix4x4<T> operator- () const;

		// binary operator add
		inline Matrix4x4<T> operator+ (const Matrix4x4<T>&) const;
		// binary operator add
		inline Matrix4x4<T> operator+ (T) const;
		// binary operator sub
		inline Matrix4x4<T> operator- (const Matrix4x4<T>&) const;
		// binary operator sub
		inline Matrix4x4<T> operator- (T) const;
		// binary operator mult
		inline Matrix4x4<T> operator* (const Matrix4x4<T>&) const;
		// binary operator mult
		inline Vector3D<T> operator* (const Vector3D<T>&) const;
		// binary operator mult
		inline Matrix4x4<T> operator* (T) const;
		// binary operator div
		inline Matrix4x4<T> operator/ (T) const;

		//! access operator
		inline T& operator() ( unsigned int i, unsigned int j ) { 
			return value[i][j];
		}
		//! constant access operator
		inline const T& operator() ( unsigned int i, unsigned int j ) const {
			return value[i][j];
		}

		// init function
		inline void initZero();
		//! init identity matrix
		inline void initId();
		//! init translation matrix
		inline void initTranslation(T x, T y, T z);
		//! init rotation matrix
		inline void initRotationX(T rot);
		inline void initRotationY(T rot);
		inline void initRotationZ(T rot);
		inline void initRotationXYZ(T rotx,T roty, T rotz);
		//! init scaling matrix
		inline void initScaling(T scale);
		inline void initScaling(T x, T y, T z);

		//! from 16 value array (init id if all 0)
		inline void initFromArray(T *array);

		inline void transpose();
		inline T	trace() const;
		inline bool invert();

		//! value access
		inline T	coeff(int i, int j) const;

		//! decompose matrix intro translation, scale, rot and shear
		void decompose(Vector3D<T> &trans, Vector3D<T> &scale, Vector3D<T> &rot, Vector3D<T> &shear);

		
		//! public to avoid [][] operators
		T value[4][4];  //< Storage of maxtrix values 

	protected:

};



//------------------------------------------------------------------------------
// STREAM FUNCTIONS
//------------------------------------------------------------------------------



/*************************************************************************
  Outputs the object in human readable form using the format
  [x,y,z]
  */
template<class T>
std::ostream&
operator<<( std::ostream& os, const Matrix4x4<T>& m )
{
	for(int i=0; i<4; i++) {
  	os << '|' << m.value[i][0] << ", " << m.value[i][1] << ", " << m.value[i][2] << ", " << m.value[i][3] << '|';
	}
  return os;
}



/*************************************************************************
  Reads the contents of the object from a stream using the same format
  as the output operator.
  */
template<class T>
std::istream&
operator>>( std::istream& is, Matrix4x4<T>& m )
{
  char c;
  char dummy[3];
  
	for(int i=0; i<4; i++) {
  	is >> c >> m.value[i][0] >> dummy >> m.value[i][1] >> dummy >> m.value[i][2] >> dummy >> m.value[i][3] >> c;
	}
  return is;
}


//------------------------------------------------------------------------------
// matrix inline FUNCTIONS
//------------------------------------------------------------------------------



/*************************************************************************
  Constructor.
  */
template<class T>
inline Matrix4x4<T>::Matrix4x4( void )
{
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {
  		value[i][j] = 0.0;
		}
	}
}



/*************************************************************************
  Copy-Constructor.
  */
template<class T>
inline Matrix4x4<T>::Matrix4x4( const Matrix4x4<T> &v )
{
  memcpy(this->value, v.value, 4*4*sizeof(T));
}

template<class T>
inline Matrix4x4<T>::Matrix4x4( const T* v )
{
    memcpy(this->value, v, 4*4*sizeof(T));
}



/*************************************************************************
  Constructor for a vector from a single T. All components of
  the vector get the same value.
  \param s The value to set
  \return The new vector
  */
template<class T>
inline Matrix4x4<T>::Matrix4x4(T s )
{
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {
  		value[i][j] = s;
		}
	}
}

/*************************************************************************
Constructor for a vector from 3 Vector3D<T>.
each COLUMN of the vector gets one Vector3D<T> value.
\the rest are set as I
\return The new vector
*/
template<class T>
inline Matrix4x4<T>::Matrix4x4(Vector3D<T> dirX, Vector3D<T> dirY, Vector3D<T> dirZ)
{
	this->initId();
	for (int i = 0; i < 3; i++){
		value[i][0] = dirX[i];
		value[i][1] = dirY[i];
		value[i][2] = dirZ[i];
	}
}



/*************************************************************************
  Copy a Matrix4x4 componentwise.
  \param v vector with values to be copied
  \return Reference to self
  */
template<class T>
inline const Matrix4x4<T>&
Matrix4x4<T>::operator=( const Matrix4x4<T> &v )
{
  value[0][0] = v.value[0][0]; value[0][1] = v.value[0][1]; value[0][2] = v.value[0][2]; value[0][3] = v.value[0][3];
  value[1][0] = v.value[1][0]; value[1][1] = v.value[1][1]; value[1][2] = v.value[1][2]; value[1][3] = v.value[1][3];
  value[2][0] = v.value[2][0]; value[2][1] = v.value[2][1]; value[2][2] = v.value[2][2]; value[2][3] = v.value[2][3];
  value[3][0] = v.value[3][0]; value[3][1] = v.value[3][1]; value[3][2] = v.value[3][2]; value[3][3] = v.value[3][3];
  return *this;
}


/*************************************************************************
  Copy a T to each component.
  \param s The value to copy
  \return Reference to self
  */
template<class T>
inline const Matrix4x4<T>&
Matrix4x4<T>::operator=(T s)
{
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {
  		value[i][j] = s;
		}
	}
  return *this;
}


/*************************************************************************
  Add another Matrix4x4 componentwise.
  \param v vector with values to be added
  \return Reference to self
  */
template<class T>
inline const Matrix4x4<T>&
Matrix4x4<T>::operator+=( const Matrix4x4<T> &v )
{
  value[0][0] += v.value[0][0]; value[0][1] += v.value[0][1]; value[0][2] += v.value[0][2]; value[0][3] += v.value[0][3];
  value[1][0] += v.value[1][0]; value[1][1] += v.value[1][1]; value[1][2] += v.value[1][2]; value[1][3] += v.value[1][3];
  value[2][0] += v.value[2][0]; value[2][1] += v.value[2][1]; value[2][2] += v.value[2][2]; value[2][3] += v.value[2][3];
  value[3][0] += v.value[3][0]; value[3][1] += v.value[3][1]; value[3][2] += v.value[3][2]; value[3][3] += v.value[3][3];
  return *this;
}


/*************************************************************************
  Add a T value to each component.
  \param s Value to add
  \return Reference to self
  */
template<class T>
inline const Matrix4x4<T>&
Matrix4x4<T>::operator+=(T s)
{
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {
  		value[i][j] += s;
		}
	}
  return *this;
}


/*************************************************************************
  Subtract another vector componentwise.
  \param v vector of values to subtract
  \return Reference to self
  */
template<class T>
inline const Matrix4x4<T>&
Matrix4x4<T>::operator-=( const Matrix4x4<T> &v )
{
  value[0][0] -= v.value[0][0]; value[0][1] -= v.value[0][1]; value[0][2] -= v.value[0][2]; value[0][3] -= v.value[0][3];
  value[1][0] -= v.value[1][0]; value[1][1] -= v.value[1][1]; value[1][2] -= v.value[1][2]; value[1][3] -= v.value[1][3];
  value[2][0] -= v.value[2][0]; value[2][1] -= v.value[2][1]; value[2][2] -= v.value[2][2]; value[2][3] -= v.value[2][3];
  value[3][0] -= v.value[3][0]; value[3][1] -= v.value[3][1]; value[3][2] -= v.value[3][2]; value[3][3] -= v.value[3][3];
  return *this;
}


/*************************************************************************
  Subtract a T value from each component.
  \param s Value to subtract
  \return Reference to self
  */
template<class T>
inline const Matrix4x4<T>&
Matrix4x4<T>::operator-=(T s)
{
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {
  		value[i][j] -= s;
		}
	}
  return *this;
}


/*************************************************************************
  Multiply with another vector component-wise.
  \param v vector of values to multiply with
  \return Reference to self
  */
template<class T>
inline const Matrix4x4<T>&
Matrix4x4<T>::operator*=( const Matrix4x4<T> &v )
{
	Matrix4x4<T> nv(0.0);
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {

			for(int k=0;k<4;k++)
				nv.value[i][j] += (value[i][k] * v.value[k][j]);
		}
	}
  *this = nv;
  return *this;
}


/*************************************************************************
  Multiply each component with a T value.
  \param s Value to multiply with
  \return Reference to self
  */
template<class T>
inline const Matrix4x4<T>&
Matrix4x4<T>::operator*=(T s)
{
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {
  		value[i][j] *= s;
		}
	}
  return *this;
}



/*************************************************************************
  Divide each component by a T value.
  \param s Value to divide by
  \return Reference to self
  */
template<class T>
inline const Matrix4x4<T>&
Matrix4x4<T>::operator/=(T s)
{
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {
  		value[i][j] /= s;
		}
	}
  return *this;
}


//------------------------------------------------------------------------------
// unary operators
//------------------------------------------------------------------------------


/*************************************************************************
  Build componentwise the negative this vector.
  \return The new (negative) vector
  */
template<class T>
inline Matrix4x4<T>
Matrix4x4<T>::operator-() const
{
	Matrix4x4<T> nv;
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {
  		nv.value[i][j] = -value[i][j];
		}
	}
  return nv;
}



//------------------------------------------------------------------------------
// binary operators
//------------------------------------------------------------------------------


/*************************************************************************
  Build a vector with another vector added componentwise.
  \param v The second vector to add
  \return The sum vector
  */
template<class T>
inline Matrix4x4<T>
Matrix4x4<T>::operator+( const Matrix4x4<T> &v ) const
{
	Matrix4x4<T> nv;
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {
  		nv.value[i][j] = value[i][j] + v.value[i][j];
		}
	}
  return nv;
}


/*************************************************************************
  Build a vector with a T value added to each component.
  \param s The T value to add
  \return The sum vector
  */
template<class T>
inline Matrix4x4<T>
Matrix4x4<T>::operator+(T s) const
{
	Matrix4x4<T> nv;
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {
  		nv.value[i][j] = value[i][j] + s;
		}
	}
  return nv;
}


/*************************************************************************
  Build a vector with another vector subtracted componentwise.
  \param v The second vector to subtract
  \return The difference vector
  */
template<class T>
inline Matrix4x4<T>
Matrix4x4<T>::operator-( const Matrix4x4<T> &v ) const
{
	Matrix4x4<T> nv;
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {
  		nv.value[i][j] = value[i][j] - v.value[i][j];
		}
	}
  return nv;
}


/*************************************************************************
  Build a vector with a T value subtracted componentwise.
  \param s The T value to subtract
  \return The difference vector
  */
template<class T>
inline Matrix4x4<T>
Matrix4x4<T>::operator-(T s ) const
{
	Matrix4x4<T> nv;
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {
  		nv.value[i][j] = value[i][j] - s;
		}
	}
  return nv;
}



/*************************************************************************
  Build a Matrix4x4 with a T value multiplied to each component.
  \param s The T value to multiply with
  \return The product vector
  */
template<class T>
inline Matrix4x4<T>
Matrix4x4<T>::operator*(T s) const
{
	Matrix4x4<T> nv;
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {
  		nv.value[i][j] = value[i][j] * s;
		}
	}
  return nv;
}




/*************************************************************************
  Build a vector divided componentwise by a T value.
  \param s The T value to divide by
  \return The ratio vector
  */
template<class T>
inline Matrix4x4<T>
Matrix4x4<T>::operator/(T s) const
{
	Matrix4x4<T> nv;
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {
  		nv.value[i][j] = value[i][j] / s;
		}
	}
  return nv;
}





/*************************************************************************
  Build a vector with another vector multiplied by componentwise.
  \param v The second vector to muliply with
  \return The product vector
  */
template<class T>
inline Matrix4x4<T>
Matrix4x4<T>::operator*( const Matrix4x4<T>& v) const
{
	Matrix4x4<T> nv(0.0);
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {

			for(int k=0;k<4;k++)
				nv.value[i][j] += (value[i][k] * v.value[k][j]);
		}
	}
  return nv;
}


template<class T>
inline Vector3D<T>
Matrix4x4<T>::operator*( const Vector3D<T>& v) const
{
	Vector3D<T> nvec(0.0);
	for(int i=0; i<3; i++) {
		for(int j=0; j<3; j++) {
			nvec[i] += (v[j] * value[i][j]);
		}
	}
	// assume normalized w coord
	for(int i=0; i<3; i++) {
		nvec[i] += (1.0 * value[i][3]);
	}
  return nvec;
}



//------------------------------------------------------------------------------
// Other helper functions
//------------------------------------------------------------------------------

template<class T>
inline void Matrix4x4<T>::initZero()
{
	(*this) = (T)(0.0);
}

//! init identity matrix
template<class T>
inline void Matrix4x4<T>::initId()
{
	(*this) = (T)(0.0);
	value[0][0] = 
	value[1][1] = 
	value[2][2] = 
	value[3][3] = (T)(1.0);
}

//! init rotation matrix
template<class T>
inline void Matrix4x4<T>::initTranslation(T x, T y, T z)
{
	this->initId();
	value[0][3] = x;
	value[1][3] = y;
	value[2][3] = z;
}

//! init rotation matrix
template<class T>
inline void 
Matrix4x4<T>::initRotationX(T rot)
{
	double drot = (double)(rot/360.0*2.0*M_PI);
	//? while(drot < 0.0) drot += (M_PI*2.0);

	this->initId();
	value[1][1] = (T)  cos(drot);
	value[1][2] = (T)  sin(drot);
	value[2][1] = (T)(-sin(drot));
	value[2][2] = (T)  cos(drot);
}
template<class T>
inline void 
Matrix4x4<T>::initRotationY(T rot)
{
	double drot = (double)(rot/360.0*2.0*M_PI);
	//? while(drot < 0.0) drot += (M_PI*2.0);

	this->initId();
	value[0][0] = (T)  cos(drot);
	value[0][2] = (T)(-sin(drot));
	value[2][0] = (T)  sin(drot);
	value[2][2] = (T)  cos(drot);
}
template<class T>
inline void 
Matrix4x4<T>::initRotationZ(T rot)
{
	double drot = (double)(rot/360.0*2.0*M_PI);
	//? while(drot < 0.0) drot += (M_PI*2.0);

	this->initId();
	value[0][0] = (T)  cos(drot);
	value[0][1] = (T)  sin(drot);
	value[1][0] = (T)(-sin(drot));
	value[1][1] = (T)  cos(drot);
}
template<class T>
inline void 
Matrix4x4<T>::initRotationXYZ( T rotx, T roty, T rotz)
{
	Matrix4x4<T> val;
	Matrix4x4<T> rot;
	this->initId();
	// todo Rachel, check here!
	// org
	/*rot.initRotationX(rotx);
	(*this) *= rot;
	rot.initRotationY(roty);
	(*this) *= rot;
	rot.initRotationZ(rotz);
	(*this) *= rot;
	// org */

	// blender
	rot.initRotationZ(rotz);
	(*this) *= rot;
	rot.initRotationY(roty);
	(*this) *= rot;
	rot.initRotationX(rotx);
	(*this) *= rot;
	// blender */
}

//! trace of matrix
template<class T>
inline T
Matrix4x4<T>::trace() const
{
	T result = 0;
	for (int i = 0; i < 4; i++)
		result += value[i][i];
	return result;
}

//! transpose matrix
template<class T>
inline void 
Matrix4x4<T>::transpose()
{
	for (int i=0;i<4;i++)
		for (int j=i+1;j<4;j++)
		{
			T a=value[i][j];
			value[i][j]=value[j][i];
			value[j][i]=a;
		}
}


//! value access
template<class T>
inline T
Matrix4x4<T>::coeff(int i, int j) const
{
	assertMsg(i >= 0 && i < 4 && j >= 0 && j < 4, "Matrix access out of boundary!");
	return value[i][j];
}

template<class T>
inline bool 
Matrix4x4<T>::invert()
{
    int indxc[4], indxr[4];
    int ipiv[4] = { 0, 0, 0, 0 };
    T minv[4][4];
    memcpy(minv, this->value, 4*4*sizeof(T));

    for (int i = 0; i < 4; i++) {
        int irow = -1, icol = -1;
        T big = 0.;
        // Choose pivot
        for (int j = 0; j < 4; j++) {
            if (ipiv[j] != 1) {
                for (int k = 0; k < 4; k++) {
                    if (ipiv[k] == 0) {
                        if (fabsf(minv[j][k]) >= big) {
                            big = T(fabsf(minv[j][k]));
                            irow = j;
                            icol = k;
                        }
                    }
                    else if (ipiv[k] > 1) {
						return false;
                        //debMsg("err - singular matrix (A) ! ", 1); exit(1);
					}
                }
            }
        }
        ++ipiv[icol];
        // Swap rows _irow_ and _icol_ for pivot
        if (irow != icol) {
            for (int k = 0; k < 4; ++k)
                swap(minv[irow][k], minv[icol][k]);
        }
        indxr[i] = irow;
        indxc[i] = icol;
        if (minv[icol][icol] == 0.) {
			//debMsg("err - singular matrix (B) ! ", 1); exit(1);
			return false;
		}

        // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
        T pivinv = 1.f / minv[icol][icol];
        minv[icol][icol] = 1.f;
        for (int j = 0; j < 4; j++)
            minv[icol][j] *= pivinv;

        // Subtract this row from others to zero out their columns
        for (int j = 0; j < 4; j++) {
            if (j != icol) {
                T save = minv[j][icol];
                minv[j][icol] = 0;
                for (int k = 0; k < 4; k++)
                    minv[j][k] -= minv[icol][k]*save;
            }
        }
    }
    // Swap columns to reflect permutation
    for (int j = 3; j >= 0; j--) {
        if (indxr[j] != indxc[j]) {
            for (int k = 0; k < 4; k++)
                swap(minv[k][indxr[j]], minv[k][indxc[j]]);
        }
    }
  	memcpy(this->value, minv, 4*4*sizeof(T));
	return true;
}

//! init scaling matrix
template<class T>
inline void 
Matrix4x4<T>::initScaling(T scale)
{
	this->initId();
	value[0][0] = scale;
	value[1][1] = scale;
	value[2][2] = scale;
}
//! init scaling matrix
template<class T>
inline void 
Matrix4x4<T>::initScaling(T x, T y, T z)
{
	this->initId();
	value[0][0] = x;
	value[1][1] = y;
	value[2][2] = z;
}


//! from 16 value array (init id if all 0)
template<class T>
inline void 
Matrix4x4<T>::initFromArray(T *array)
{
	bool allZero = true;
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {
			value[i][j] = array[i*4+j];
			if(array[i*4+j]!=0.0) allZero=false;
		}
	}
	if(allZero) this->initId();
}

//! decompose matrix
template<class T>
void 
Matrix4x4<T>::decompose(Vector3D<T> &trans,Vector3D<T> &scale,Vector3D<T> &rot,Vector3D<T> &shear) {
	Vec3 row[3],temp;

	for(int i = 0; i < 3; i++) {
		trans[i] = this->value[3][i];
	}

	for(int i = 0; i < 3; i++) {
		row[i][0] = this->value[i][0];
		row[i][1] = this->value[i][1];
		row[i][2] = this->value[i][2];
	}

	scale[0] = norm(row[0]);
	normalize (row[0]);

	shear[0] = dot(row[0], row[1]);
	row[1][0] = row[1][0] - shear[0]*row[0][0];
	row[1][1] = row[1][1] - shear[0]*row[0][1];
	row[1][2] = row[1][2] - shear[0]*row[0][2];

	scale[1] = norm(row[1]);
	normalize (row[1]);

	if(scale[1] != 0.0)
		shear[0] /= scale[1];

	shear[1] = dot(row[0], row[2]);
	row[2][0] = row[2][0] - shear[1]*row[0][0];
	row[2][1] = row[2][1] - shear[1]*row[0][1];
	row[2][2] = row[2][2] - shear[1]*row[0][2];

	shear[2] = dot(row[1], row[2]);
	row[2][0] = row[2][0] - shear[2]*row[1][0];
	row[2][1] = row[2][1] - shear[2]*row[1][1];
	row[2][2] = row[2][2] - shear[2]*row[1][2];

	scale[2] = norm(row[2]);
	normalize (row[2]);

	if(scale[2] != 0.0) {
		shear[1] /= scale[2];
		shear[2] /= scale[2];
	}

	temp = cross(row[1], row[2]);
	if(dot(row[0], temp) < 0.0) {
		for(int i = 0; i < 3; i++) {
			scale[i]  *= -1.0;
			row[i][0] *= -1.0;
			row[i][1] *= -1.0;
			row[i][2] *= -1.0;
		}
	}

	if(row[0][2] < -1.0) row[0][2] = -1.0;
	if(row[0][2] > +1.0) row[0][2] = +1.0;

	rot[1] = asin(-row[0][2]);

	if(fabs(cos(rot[1])) > VECTOR_EPSILON) {
		rot[0] = atan2 (row[1][2], row[2][2]);
		rot[2] = atan2 (row[0][1], row[0][0]);
	}
	else {
		rot[0] = atan2 (row[1][0], row[1][1]);
		rot[2] = 0.0;
	}

	rot[0] = (180.0/M_PI)*rot[0];
	rot[1] = (180.0/M_PI)*rot[1];
	rot[2] = (180.0/M_PI)*rot[2];
} 

//------------------------------------------------------------------------------
// TYPEDEFS
//------------------------------------------------------------------------------


typedef Matrix4x4<double>	Mat4d; 

// a 3D vector with single precision
typedef Matrix4x4<float>	Mat4f; 

// a 3D integer vector
typedef Matrix4x4<int>		Mat4i; 

// default vector typing
// a 3D vector for graphics output, typically float?
typedef Matrix4x4<Real>  Mat4; 

} // namespace

#endif




