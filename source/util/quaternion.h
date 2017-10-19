/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL) 
 * http://www.gnu.org/licenses
 *
 * Basic quaternion class
 *
 ******************************************************************************/

#ifndef _QUATERNION_H
#define _QUATERNION_H

#include "vectorbase.h"
#include "matrixbase.h"

namespace Manta {

//! Very basic quaternion class
class Quaternion {
public:
    
    //! default constructor
    Quaternion() : x(0), y(0), z(0), w(0) {}
    
    //! copy constructor
    Quaternion(const Quaternion& q) : x(q.x), y(q.y), z(q.z), w(q.w) {}
    
    //! construct a quaternion from members
    Quaternion(Real _x, Real _y, Real _z, Real _w) : x(_x), y(_y), z(_z), w(_w) {}
        
    //! construct a quaternion from imag/real parts
    Quaternion(Vec3 i, Real r) : x(i.x), y(i.y), z(i.z), w(r) {}
    
	// ! construct a quaternion from a rotation Matrix
	// This algorithm comes from  "Quaternion Calculus and Fast Animation",
	// Ken Shoemake, 1987 SIGGRAPH course notes
	Quaternion(const Mat4& mat){
		Real t = mat.coeff(0, 0) + mat.coeff(1, 1) + mat.coeff(2, 2);
		if (t > VECTOR_EPSILON){
			t = sqrt(t + mat.coeff(3, 3));
			w = 0.5f*t;
			t = 0.5f/t;
			x = (mat.coeff(2, 1) - mat.coeff(1, 2)) * t;
			y = (mat.coeff(0, 2) - mat.coeff(2, 0)) * t;
			z = (mat.coeff(1, 0) - mat.coeff(0, 1)) * t;
		}
		else{
			int i = 0;
			if (mat.coeff(1, 1) > mat.coeff(0, 0)) i = 1;
			if (mat.coeff(2, 2) > mat.coeff(i, i)) i = 2;
			int j = (i + 1) % 3;
			int k = (j + 1) % 3;

			t = sqrt(mat.coeff(i, i) - mat.coeff(j, j) - mat.coeff(k, k) + 1.0f);
			Vec3 qv(0.0f);
			qv[i] = 0.5f*t;
			t = 0.5f/t;
			qv[j]= (mat.coeff(j, i) + mat.coeff(i, j))*t;
			qv[k]= (mat.coeff(k, i) + mat.coeff(i, k))*t;
			w = (mat.coeff(k, j) - mat.coeff(j, k))*t;
			x = qv.x; y = qv.y; z = qv.z;
		}
		if (fabs(mat.coeff(3,3)-1.0f) > VECTOR_EPSILON) {
			Real s = 1.0f / sqrt(mat.coeff(3,3));
			w *= s; x *= s; y *= s; z *= s;
		}
	}

    //! Assign operator
    inline Quaternion& operator= (const Quaternion& q) {
        x = q.x;
        y = q.y;
        z = q.z;
        w = q.w;
        return *this;
    }
    
    //! Assign multiplication operator
    inline Quaternion& operator*= ( const Real a ) {
        x *= a;
        y *= a;
        z *= a;
        w *= a;
        return *this;
    }
    
    //! return inverse quaternion
    inline Quaternion inverse() const {
        Real mag = 1.0/(x*x+y*y+z*z+w*w);
        return Quaternion(-x*mag,-y*mag,-z*mag,w*mag);
    }
    
    //! imaginary part accessor
    inline Vec3 imag() { return Vec3(x,y,z); }

    // imaginary part
    Real x;
    Real y;
    Real z;
    
    // real part
    Real w;    
};


//! Multiplication operator
inline Quaternion operator* ( const Quaternion &q1, const Quaternion &q2 ) {
    return Quaternion ( q2.w * q1.x + q2.x * q1.w + q2.y * q1.z - q2.z * q1.y,
                        q2.w * q1.y + q2.y * q1.w + q2.z * q1.x - q2.x * q1.z,
                        q2.w * q1.z + q2.z * q1.w + q2.x * q1.y - q2.y * q1.x,
                        q2.w * q1.w - q2.x * q1.x - q2.y * q1.y - q2.z * q1.z );
}

//! Multiplication operator
inline Quaternion operator* ( const Quaternion &q, const Real a ) {
    return Quaternion ( q.x*a, q.y*a, q.z*a, q.w*a);
}

inline Quaternion operator+(const Quaternion &p, const Quaternion &q){
	return Quaternion(p.x + q.x, p.y + q.y, p.z + q.z, p.w + q.w); 
}

inline Quaternion operator-(const Quaternion &p, const Quaternion &q){
	return Quaternion(p.x - q.x, p.y - q.y, p.z - q.z, p.w - q.w); 
}

inline Quaternion operator-(const Quaternion &p){
	return Quaternion(-p.x, -p.y, -p.z, -p.w);
}

inline Real dot(const Quaternion &p,const Quaternion &q){
	return p.x*q.x + p.y*q.y + p.z*q.z + p.w*q.w; 
}

inline Real normSquare(const Quaternion &p) { return dot(p, p); }
inline Real norm(const Quaternion &p) {
	Real l = dot(p, p);
	return (fabs(l - 1.) < VECTOR_EPSILON*VECTOR_EPSILON) ? 1. : sqrt(l);
}

inline Quaternion getNormalized(const Quaternion &p){
	Real d = norm(p);
	if (d < VECTOR_EPSILON){
		return Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
	}
	d = 1.0f / d;
	return Quaternion(p.x*d, p.y*d, p.z*d, p.w*d);
}

inline Quaternion slerp(Quaternion q, Quaternion p, const Real t)
{//the return value is normalized, if q and p are normalized
	Real cosphi = dot(q, p);

	if (cosphi < 0.0f)
	{
		cosphi *= -1.0f;
		q = -q;//..
	}

	const Real DOT_THRESHOLD = 0.9995f;
	if (cosphi > DOT_THRESHOLD) {
		// interpolate linearly
		return getNormalized(q + (p - q) * t);
	}

	Real sinphi = sqrt(1. - cosphi * cosphi);
	Real phi = acos(cosphi);

	Quaternion res = q * (sin(phi * (1. - t)) / sinphi) + p * (sin(phi * t) / sinphi);

	return res;
}

inline void convAngleAxisN2Quat(Vec3 AxisN, Real Angle, Quaternion& q){
	q = Quaternion(AxisN * sin(Real(0.5) * Angle), cos(Real(0.5) * Angle));
}

inline void convAngleAxis2Quat(Vec3 Axis, Real Angle, Quaternion& q){
	Vec3 AxisN = getNormalized(Axis);
	convAngleAxisN2Quat(AxisN, Angle, q);
}

inline void convQuatU2AngleAxisN(Quaternion q, Vec3 &AxisN, Real& Angle){
	Real n2 = normSquare(q.imag());
	if (n2 < VECTOR_EPSILON*VECTOR_EPSILON){
		Angle = 0.0f;
		AxisN = Vec3(1.0f, 0.0f, 0.0f);
	}else{
		Real cosTheta = q.w;
		if (q.w > 1.0f) cosTheta = 1.0f;
		else if (q.w < -1.0f) cosTheta = -1.0f;
		Angle = 2.0f * acos(cosTheta);
		AxisN = q.imag() / sqrt(n2);
	}
}

inline void convQuat2AngleAxisN(Quaternion q, Vec3 &AxisN, Real& Angle){
	Quaternion qU = getNormalized(q);
	convQuatU2AngleAxisN(qU, AxisN, Angle);
}

inline Vec3 transformVecWithQuat(Quaternion q, Vec3 v)
{
	Vec3 uv = cross(q.imag(), v);
	uv += uv;
	return v + q.w * uv + cross(q.imag(), uv);
}

inline Vec3 transformVecWithAngleAxisN(Vec3 AxisN, Real Angle, Vec3 v)
{
	Quaternion tmpQ;
	convAngleAxisN2Quat(AxisN, Angle, tmpQ);
	return transformVecWithQuat(tmpQ, v);
}

} // namespace

#endif
