/******************************************************************************
*
* MantaFlow fluid solver framework
*
* Patch deformation
*
******************************************************************************/

#ifndef MANTA_TETELEMENT_H
#define MANTA_TETELEMENT_H

#include "vectorbase.h"
#include "matrixbase.h"
#include "mesh.h"

#include <vector>

namespace Manta {

extern const Vec3 stateRestUnit[];
extern const int tetIndices[];

inline float getConeWeight(float dis, float inner = 0.33333f) {// dis between [0,1]
	if (dis < inner)
		return 1.0f;
	if (dis < 1.0f)
		return (1.0f - dis) / (1.0f - inner);
	return 0.0f;
}

struct Patch_RestInfo {
	//data 
	//! current tets, identical to stateRestUnit and tetIndices in the simplest case
	int subdN;
	std::vector<int>  currIndices;
	std::vector<Vec3> restPoints;
	std::vector<Mat4> defoInvRest;
	
	// convenience/redundant - has to match currStateRest.size and currIndices.size
	std::vector<Vec3> restTets;
	void* HS; // avoid include eigen header every where...
	int currSize;
	int currPoints;

	// init
	void init(int subdiv);
	Patch_RestInfo(int subdiv) {
		HS = NULL;
		init(subdiv);		
	}

	~Patch_RestInfo();
};

inline int isInsideOpt(const std::vector<Vec3>& stateDeformed, const std::vector<Mat4>& defoInv, const Vec3& p,
	const Vec3& bbmin, const Vec3& bbmax, Vec3* retBary)
{
	if ((p.x<bbmin.x) ||
		(p.y<bbmin.y) ||
		(p.z<bbmin.z) ||
		(p.x>bbmax.x) ||
		(p.y>bbmax.y) ||
		(p.z>bbmax.z)) return -8;

	for (int tet = 0; tet < defoInv.size(); tet++) {
		if (defoInv[tet](0, 0) == -1000.) continue;

		Vec3 pTr = p - stateDeformed[4 * tet];
		Vec3 bary = defoInv[tet] * pTr;
		// NT_DEBUG, note optimization possible: merge offset into matrix, do early reject after each componet?
		Real baryWeight4 = (1. - bary[0] - bary[1] - bary[2]);

		//debMsg("AAB is in? "<<bary<<" "<<p<<"  "<<tet ,1);
		if ((bary[0]     < (0.f - VECTOR_EPSILON)) || (bary[0]     > (1.f + VECTOR_EPSILON)) || (bary[1] < (0.f - VECTOR_EPSILON)) ||
			(bary[1]     > (1.f + VECTOR_EPSILON)) || (bary[2]     < (0.f - VECTOR_EPSILON)) || (bary[2] > (1.f + VECTOR_EPSILON)) ||
			(baryWeight4 < (0.f - VECTOR_EPSILON)) || (baryWeight4 >(1.f + VECTOR_EPSILON))) {
			// not inside, continue searching...
		}
		else {
			if (retBary) *retBary = bary;
			return 4 * tet;
		}
	}

	// not found...
	return -8;
}

//! small class to hold defo data
class Patch_DefoData {
public:
	Patch_DefoData() : points(), tets(), bbmin(1e10), bbmax(-1e10), restInfo(NULL) { }
	~Patch_DefoData() {};

	void init(Patch_RestInfo& rest, std::vector<Vec3> *_points = NULL); // init with defo
	
	int isInside(const Vec3& p, Vec3* retBary) const {
		return isInsideOpt(tets, defoInv, p, bbmin, bbmax, retBary);
	}
	
	// position transform funtions
	// assumes pPatch position in normalized local coords [0..1], hintTetID used for acceleration
	Vec3 local2World(const Vec3& pPatch, int * hintTetID = NULL) const ;
	// return position in normalized local coords [0..1], hintTetID used for acceleration
	Vec3 world2Local(const Vec3& pWorld, bool* isInside = NULL, int * hintTetID = NULL) const;

	// vector transform functions, pPatch position in local coords [0..1], hintTetID used for acceleration
	// local vector to world, undeformed->deformed, only change direction!
	Vec3 localVec2WorldVec(const Vec3& pPatch, const Vec3& vPatch, int * hintTetID = NULL) const;
	// world vector to local, deformed->undeformed, only change direction!
	Vec3 worldVec2LocalVec(const Vec3& pPatch, const Vec3& vWorld, int * hintTetID = NULL) const;

	// deformation Control functions
	Real adjustDefoInPlace(std::vector<Vec3> & vertex, Real lamda = 0.0f);
	void adjustScaleInPlace(Real patchLen, Vec3& oldC);

	// may have bugs... unused
	void cageAvgBase(Vec3& dirX, Vec3& dirY, Vec3& dirZ);
	void cageAvgBaseLen(Real& dirXL, Real& dirYL, Real& dirZL);

	// get|set functions
	inline bool hasDefo() { return (points.size() > 0); }
	inline Vec3 getPos(int vi) { return points[vi];	}
	inline void setPos(int vi, Vec3& p) { points[vi] = p; }

	inline bool inbbCheck(Vec3& p) {
		if ((p.x<bbmin.x) || (p.y<bbmin.y) || (p.z<bbmin.z) ||
			(p.x>bbmax.x) || (p.y>bbmax.y) || (p.z>bbmax.z))
			return false;
		return true;
	}

	// just for visualization
	void meshView(Mesh& mesh, Vec3 factor = Vec3(1.0));
protected:
	// not explicitly declared values: 
	// cage subd number n, cell number = n^3, vertex number m = (n+1)^3, n = (restInfo->subdN + 1)

	//! cage vertex in world space
	std::vector<Vec3> points; // points.size() =  m = (n+1)^3
	//! concatenated tet vertices, not shared! (5 tets for one cell)
	std::vector<Vec3> tets;  // tets.size() = 5 * n^3
	//! inverese tet deformations
	std::vector<Mat4> defoInv;  // defoInv.size() = 5 * n^3
	//! world space bounding box for quad points, little acceleration
	Vec3 bbmin, bbmax;

	Patch_RestInfo* restInfo;
};



}

#endif