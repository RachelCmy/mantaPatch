/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL) 
 * http://www.gnu.org/licenses
 *
 * Deformation-limiting Patch Advection by Mengyu Chu (mengyu.chu@tum.de)
 *
 * This is an implementation of deformation-limiting patch motion [1].
 * The algorithm always works with 3D cube cages. In 2D, the front xy plane
 * is used.
 *
 * [1] Data-Driven Synthesis of Smoke Flows with CNN-based Feature Descriptors,
 *     Mengyu Chu and Nils Thuerey. 2017. ACM Trans. Graph. 36, 4 (2017).
 *
 ******************************************************************************/
 
#ifndef PATCHADV_H
#define PATCHADV_H

#include "vectorbase.h"
#include "grid.h"
#include "particle.h"

#include "defoElement.h"
//#include "quaternion.h"
//#include <cmath>

#ifndef WIN32
#ifndef sprintf_s 
// sprintf_s is windows specific
#define sprintf_s sprintf
#endif
#endif

namespace Manta {

class Mesh;

class PatchData {
public:
//protected:
	Vec3	pos;
	int		flag;
	int		uniqID;		// which is the unique patch ID when saveing local patch data
	int		lifet;		// which is the frame ID when saveing local patch data

	Real	fadeWei;	// temporal fading weights

	Patch_DefoData	defoData;	// defo data
	Real			defoError;	// defo errir
public:
	PatchData() : pos(0.0f), flag(0), lifet(0), \
		fadeWei(1.0f), defoData(), defoError(0.0f), uniqID(-1) {}
	PatchData(const Vec3& p, Real fadeW = 1.0, int unID = -1) : pos(p), flag(0), lifet(0), \
		fadeWei(fadeW), defoData(), defoError(0.0f), uniqID(unID) {}

	void initDefoPos(const Vec3& dirX, const Vec3& dirY, const Real reslen, Patch_RestInfo& rest) {
		Vec3 dirZ = cross(dirX, dirY);
		normalize(dirZ);
		std::vector<Vec3> points(rest.currPoints);
		
		for (int vi = 0; vi < rest.currPoints ; ++vi) {// [-0.5, 0.5]
			points[vi] = dirX * (rest.restPoints[vi].x - 0.5) +
				dirY * (rest.restPoints[vi].y - 0.5) +
				dirZ * (rest.restPoints[vi].z - 0.5);
			points[vi] *= reslen;
			points[vi] += pos;
		}
		defoData.init(rest, &points);
	}

	static ParticleBase::SystemType getType() { return ParticleBase::PARTICLE; }
};

template class ParticleSystem<PatchData>;

/*! Basetype S must at least contain all things in PatchData */
PYTHON() template<class S>
class PatchSystem : public ParticleSystem<S> {
public:
	enum PatchStatus {
		PNONE = ParticleBase::ParticleStatus::PNONE,
		PNEW = ParticleBase::ParticleStatus::PNEW,  // particles newly created in this step
		PDELETE = ParticleBase::ParticleStatus::PDELETE, // mark as deleted, will be deleted in next compress() step
		PINVALID = ParticleBase::ParticleStatus::PINVALID, // unused
		PFADINGIN = (1 << 3),// mark as fading in, will become normal patch afterwards
		PFADINGOUT = (1 << 4)// mark as fading out, will be deleted afterwards		
	};

	// Parent functions
	inline bool isActive(IndexInt idx) const { return ParticleSystem<S>::isActive(idx);	}
	inline void kill(IndexInt idx) { ParticleSystem<S>::kill(idx);	}
	inline bool isFadingIn(IndexInt idx) const {
		DEBUG_ONLY(this->checkPartIndex(idx));
		return (mData[idx].flag & PatchStatus::PFADINGIN) != 0;
	}
	inline bool isNewPatch(IndexInt idx) const {
		DEBUG_ONLY(this->checkPartIndex(idx));
		return (mData[idx].flag & PatchStatus::PNEW) != 0;
	}
	inline bool isFadingOut(IndexInt idx) const {
		DEBUG_ONLY(this->checkPartIndex(idx));
		return (mData[idx].flag & PatchStatus::PFADINGOUT) != 0;
	}
	FluidSolver* getParent() const { return PbClass::getParent(); }
	inline IndexInt size() const { return ParticleSystem<S>::size(); }
	void doCompress() { ParticleSystem<S>::doCompress(); }
	IndexInt add(const S& data) { 
		IndexInt resultID = ParticleSystem<S>::add(data); 
		mData[resultID].flag |= PNEW;
		mData[resultID].uniqID = (pCount++);
		//debMsg("add funtion, "<<pCount, 0);
		return resultID;
	}

	// ----- PYTHON functions ----------------------------------------------------
	PYTHON() PatchSystem(FluidSolver* parent, int subdiv = 2, int baseRes = 12, int jN = 1)
		: ParticleSystem<S>(parent), patRest(subdiv), patResoLen(baseRes), jitN(jN),
		pNeib(NULL), pCount(0), cellList(NULL), tetList(NULL), preIndexGrid(NULL), preTetGrid(NULL), preSolver(NULL)
	{
		/* // debug!!
		Patch_DefoData testData;
		testData.init(patRest, &(patRest.restPoints));
		std::vector<Vec3> Newp(patRest.restPoints);
		for (int i = 0; i < Newp.size(); ++i) {
			Newp[i] = patRest.restPoints[i] * 2.0 + Vec3(0.5, 0.5, 0.5);
		}
		testData.adjustDefoInPlace(Newp,0.002);
		*/
	}
	~PatchSystem() {
		if (pNeib)			delete pNeib;
		if (preIndexGrid)	delete preIndexGrid;
		if (preTetGrid)		delete preTetGrid;
		if (preSolver)		delete preSolver;
		if (cellList)		delete[] cellList;
		if (tetList)		delete[] tetList;
	}

	PYTHON() int addPatch(Vec3 pos, Real fadeW = 1.0) {// without parallelization
		return add( PatchData(pos, fadeW) );
	}

	PYTHON() void initPatchCage3D(int idx, const Vec3& dirX, const Vec3& dirY) { // for 3D
		mData[idx].initDefoPos(dirX, dirY, patResoLen, patRest);
	}

	PYTHON() void initPatchCage2D(int idx, const Vec3& dirX) { // for 2D
		Vec3 dirY = cross(Vec3(0.0,0.0,1.0), dirX);
		normalize(dirY);
		mData[idx].initDefoPos(dirX, dirY, patResoLen, patRest);
	}

	// generate to candidate list
	PYTHON() void sampleCandidates(const Grid<Real>& denG, Real samWidth, Real weiThresh = 1.0, 
		Real occThresh = 0.5, Grid<Real>* patchWeiGp = NULL);

	PYTHON() void initCandidateCage(Grid<Real>& denG);

	PYTHON() void addCandidatePatch();
	//PYTHON() void initCandidateParam();
	// have to use lamda instead of lambda, since lambda is a keyword in python
	PYTHON() void AdvectWithControl(const Real lamda, FlagGrid& flags, const MACGrid& vel, const int integrationMode, const bool scaleLen = false);
	// tarSZ is the final resolution we want to synthesis, tarG.size()
	PYTHON() void synPerCellAcceleration(Vec3i tarSZ = Vec3i(0));
	PYTHON() void patchSynthesisReal( Grid<Real>& tarG, Vec3i patchSZ, PyArrayContainer patchGrids,
		PyArrayContainer* patchDict = NULL, bool withSpaceW = true, Grid<Real>* weigG = NULL, bool clear = true);

	PYTHON() void patchSynthesisMAC(MACGrid& tarG, Vec3i patchSZ, PyArrayContainer patchGrids,
		PyArrayContainer* patchDict = NULL, bool withSpaceW = true, Grid<Real>* weigG = NULL);

	// save cage field into grid (undeformed), and save to disk / numpy
	// accSZ decides the size of the acceleration data structure.
	PYTHON() void saveLocalPerCellAcceleration(Vec3i accSZ = Vec3i(0));
	PYTHON() void saveLocalPatchGridReal(Grid<Real>& dataG, std::string dirPath, bool rescale = false, Real* step = NULL, bool ppmFlag = false, bool doFadOut = true);
	PYTHON() void saveLocalPatchMACGrid( MACGrid& dataG, std::string dirPath, bool rescale = false, Real* step = NULL, bool ppmFlag = false, bool doFadOut = true);
	PYTHON() void saveLocalPatchMACCurl( MACGrid& dataG, std::string dirPath, bool rescale = false, Real* step = NULL, bool ppmFlag = false, bool doFadOut = true);
	//// functions are very similar, todo reuse code properly.
	PYTHON() int saveLocalPatchNumpyReal(Grid<Real>& dataG, PyArrayContainer patchGrids, PyArrayContainer patchDict, Vec3i desRegion = Vec3i(0), bool doFadOut = false);
	PYTHON() int saveLocalPatchNumpyMAC(MACGrid& dataG, PyArrayContainer patchGrids, PyArrayContainer patchDict, Vec3i desRegion = Vec3i(0), bool doFadOut = false);
	PYTHON() int saveLocalPatchNumpyCurl(MACGrid& dataG, PyArrayContainer patchGrids, PyArrayContainer patchDict, Vec3i desRegion = Vec3i(0), bool doFadOut = false);
	PYTHON() void getCagePosParSys(BasicParticleSystem& target);

	// todo test saveGrid by synthesis
	//PYTHON() void deletBadPatches();
	PYTHON() void updateParts(bool compress = false);
	PYTHON() void killBad(Real maxDefE, PyArrayContainer* BadList = NULL, Grid<Real>* den = NULL);
	// visualization
	PYTHON() void meshView(Mesh& mesh, Vec3 factor = Vec3(1.0));

	
    // ----- get|set functions ----------------------------------------------------
	void addParLifeT(int idx) { DEBUG_ONLY(checkPartIndex(idx));++(mData[idx].lifet); }
	void removeNewFlag(int idx) { DEBUG_ONLY(checkPartIndex(idx));
		if ((mData[idx].flag & PNEW) != 0) mData[idx].flag -= PNEW;
	}
	const Real getFading(int idx) const { return mData[idx].fadeWei; }
	void setFading(int idx, Real fadW) { mData[idx].fadeWei = fadW; }
	void setFadingIn(int idx) { this->mData[idx].flag |= PatchStatus::PFADINGIN; }

	Vec3 getRestPDDPos(int vi) {
		return patRest.restPoints[vi];
	}

	Vec3 patlocal2World(int idx, const Vec3& pPatch, int * hintTetID = NULL) {
		Vec3 localp(pPatch);
		bool is2D = getParent()->is2D();
		if (is2D) localp.z = 0.5f;
		Vec3 worldp = mData[idx].defoData.local2World(localp, hintTetID);
		if (is2D) worldp.z = 0.5f;
		return worldp;
	}
	// return position in normalized local coords [0..1], hintTetID used for acceleration
	Vec3 world2patLocal(int idx, const Vec3& pWorld, bool* isInside = NULL, int * hintTetID = NULL) {
		Vec3 worldp(pWorld);
		bool is2D = getParent()->is2D();
		if (is2D) worldp.z = 0.5 * patResoLen;
		Vec3 localp = mData[idx].defoData.world2Local(worldp, isInside, hintTetID);
		if (is2D) localp.z = 0.5f;
		return localp;
	}

	// vector transform functions, pPatch position in local coords [0..1], hintTetID used for acceleration
	// local vector to world, undeformed->deformed
	Vec3 patlocalVec2WorldVec(int idx, const Vec3& pPatch, const Vec3& vPatch, int * hintTetID = NULL) {
		Vec3 localp(pPatch);
		bool is2D = getParent()->is2D();
		if (is2D) localp.z = 0.5f;
		return mData[idx].defoData.localVec2WorldVec(localp, vPatch, hintTetID);
	}
	// world vector to local, deformed->undeformed
	Vec3 worldVec2patLocalVec(int idx, const Vec3& pPatch, const Vec3& vWorld, int * hintTetID = NULL) {
		Vec3 localp(pPatch);
		bool is2D = getParent()->is2D();
		if (is2D) localp.z = 0.5f;
		return mData[idx].defoData.worldVec2LocalVec(localp, vWorld, hintTetID);
	}

	Vec3 getDefoCornerPos(IndexInt idx, int vi) {
		DEBUG_ONLY(checkPartIndex(idx));
		DEBUG_ONLY(assertMsg(vi < patRest.currPoints, "index out of boundary!"));
		return mData[idx].defoData.getPos(vi);
	}

	Vec3 getDefoPos(int idx, int preidx, Vec3& uvw) {
		Real uv4 = (1. - uvw[0] - uvw[1] - uvw[2]);
		Vec3 smokepos = mData[idx].defoData.getPos(patRest.currIndices[4 * preidx + 0]) * uv4 +
			mData[idx].defoData.getPos(patRest.currIndices[4 * preidx + 1]) * uvw.x +
			mData[idx].defoData.getPos(patRest.currIndices[4 * preidx + 2]) * uvw.y +
			mData[idx].defoData.getPos(patRest.currIndices[4 * preidx + 3]) * uvw.z;

		if (getParent()->is2D()) smokepos.z = 0.5f;
		return smokepos;
	}

	// deformation control functions
	void adjustDefo(int idx, std::vector<Vec3> & vertex, Real lambda = 0.0f) {
		mData[idx].defoError = mData[idx].defoData.adjustDefoInPlace(vertex, lambda);
	}

	void adjustScale(int idx) {
		mData[idx].defoData.adjustScaleInPlace(patResoLen, mData[idx].pos);
	}

	void setCentrePos(int idx) {
		Vec3 centrePos(0.0);
		Real centreWei = 0;//todo weight
		bool is2D = getParent()->is2D();
		int vn = patRest.restPoints.size();
		if (is2D) vn /= (2 + patRest.subdN);

		for (int vi = 0; vi < vn; ++vi) {
			Vec3 curpos = patRest.restPoints[vi];
			if (is2D) curpos.z = 0.5f;
			float curWei = getConeWeight(/*2.0f**/norm(curpos - Vec3(.5)));// do not want 0 for boundary points, always positive wei

			centrePos += mData[idx].defoData.getPos(vi) * curWei;
			centreWei += curWei;
		}
		Vec3 targetPos = centrePos / centreWei;
		if (is2D) targetPos.z = 0.5f;
		mData[idx].pos = targetPos;
	}

	void scaleCage(int idx) {// has bugs!!!
		Real dirX, dirY, dirZ;
		mData[idx].defoData.cageAvgBaseLen(dirX, dirY, dirZ);
		Vec3 factor(dirX, dirY, dirZ);
		factor = factor / (patResoLen / (patRest.subdN + 1));
		//factor.x = clamp(factor.x, 0.9f, 1.1f);
		//factor.y = clamp(factor.y, 0.9f, 1.1f);
		//factor.z = clamp(factor.z, 0.9f, 1.1f);
		int vn = patRest.restPoints.size();
		for (int vi = 0; vi < vn; ++vi) {
			Vec3 newpos = (mData[idx].defoData.getPos(vi) - mData[idx].pos) / factor + mData[idx].pos;
			mData[idx].defoData.setPos(vi, newpos);
		}
	}

	void resetDefoMatrix(int idx) {
		mData[idx].defoData.init(patRest);
	}

	
public:
	// params
	int		jitN;		// sampling candidate number
	Real	patResoLen; // base patch resolution

protected:
	// permanent tools:
	// tools:
	Patch_RestInfo	patRest; // the rest shape and other info about the deformable cage
	int				pCount; // start from 0, increase when new patch created, never decrease, used as unique ID

	// temporary tools:
	// tool for sampling
	std::vector<S> candpos_list;
	// a list for candidates, should select good ones and add into mData
	// avoid unique IDs of patches increasing unnecessarily

	// tool for advection
	BasicParticleSystem* pNeib;

	// tools for acceleration according to current particle distribution and deformation
	// per-cell acceleration for world2patch position transfer, used in synthesis
	// need to rebuild when patches advected/changed
	std::vector<int> * cellList;
	std::vector<int> * tetList;
	// used in save local patch grids
	Grid<int>* preIndexGrid;
	Grid<Vec3>* preTetGrid;
	FluidSolver* preSolver;

	using ParticleSystem<PatchData>::mData;
};

PYTHON() alias PatchSystem<PatchData> PatchAdvSys;

// blur grid functions and SAT functions
KERNEL() template<class T>
void knGetGridSAT1D(Grid<T>& sourceGrid, int index = 0) {
	Vec3i sz = sourceGrid.getSize();
	if (index == 0 && i != 0) return;
	if (index == 1 && j != 0) return;
	if (index == 2 && k != 0) return;
	// non parallel part
	for (int cell = 1; cell < (sz[index]); ++cell) {
		Vec3i prePos(i, j, k), curPos(i, j, k);
		prePos[index] = cell - 1;
		curPos[index] = cell;
		sourceGrid(curPos) += sourceGrid(prePos);
	}

}
template<class T>
void getGridSAT(Grid<T>& sourceGrid, Grid<T>& tarGrid ) {
	tarGrid.copyFrom(sourceGrid);
	knGetGridSAT1D<T>(tarGrid, 0);
	knGetGridSAT1D<T>(tarGrid, 1);
	knGetGridSAT1D<T>(tarGrid, 2);
}

template<class T>
T getSumFromSAT(Grid<T>& SATGrid, Vec3i& bgP1, Vec3i& edP) {
	if( !SATGrid.isInBounds(edP) || !SATGrid.isInBounds(bgP1))
		return T(0.0);

	for (int index = 0; index < 3; index++)
		if (edP[index] < bgP1[index])
			return T(0.0);

	Vec3i bgP = bgP1 - Vec3i(1, 1, 1);
	T result = SATGrid(edP);
	if (bgP.x >= 0)
		result -= SATGrid(bgP.x, edP.y, edP.z);
	if (bgP.y >= 0)
		result -= SATGrid(edP.x, bgP.y, edP.z);
	if (bgP.z >= 0)
		result -= SATGrid(edP.x, edP.y, bgP.z);
	if (bgP.x >= 0 && bgP.y >= 0)
		result += SATGrid(bgP.x, bgP.y, edP.z);
	if (bgP.x >= 0 && bgP.z >= 0)
		result += SATGrid(bgP.x, edP.y, bgP.z);
	if (bgP.y >= 0 && bgP.z >= 0)
		result += SATGrid(edP.x, bgP.y, bgP.z);
	if (bgP.x >= 0 && bgP.y >= 0 && bgP.z >= 0)
		result -= SATGrid(bgP);
	return result;
}


KERNEL() template<class T>
void knConvGrid1D(const Grid<T>& source, Grid<T>& target, std::vector<Real>& kern1d, int index = 0){
	int ksz = kern1d.size();
	int c = ksz / 2;
	Vec3i pos(i, j, k), step(0, 0, 0);
	step[index] = 1;

	T pxResult = source(pos) * kern1d[c];

	for (int ci = 1; ci <= c; ci++) {
		Vec3i curpos = Vec3i(i, j, k) - step * ci;
		if (!source.isInBounds(curpos)) {
			curpos.x = clamp(curpos.x, 0, source.getSizeX()-1);
			curpos.y = clamp(curpos.y, 0, source.getSizeY()-1);
			curpos.z = clamp(curpos.z, 0, source.getSizeZ()-1);
		}
		pxResult = pxResult + kern1d[c - ci] * source.get(curpos);

		curpos = Vec3i(i, j, k) + step * ci;
		if (!source.isInBounds(curpos)) {
			curpos.x = clamp(curpos.x, 0, source.getSizeX()-1);
			curpos.y = clamp(curpos.y, 0, source.getSizeY()-1);
			curpos.z = clamp(curpos.z, 0, source.getSizeZ()-1);
		}
		pxResult = pxResult +  kern1d[c + ci] * source.get(curpos);
	}
	
	target(pos) = pxResult;
}

inline void get1DKernel(std::vector<Real>& kern, int& ksz, Real sigm = -0.1) {
	// adjust params properly
	if (ksz % 2 == 0) ksz += 1;
	if (sigm < VECTOR_EPSILON) sigm = (Real) ksz / 6.0;

	Real m = 1.0 / (sqrt(2.0 * M_PI) * sigm), s2 = sigm * sigm;
	int c = ksz / 2;
	kern.resize(ksz);

	Real vall = kern[c] = m;
	for (int i = 1; i <= c; i++) {
		float v = m * exp(-(1.0*i*i) / (2.0 * s2));
		kern[c + i] = v;
		kern[c - i] = v;
		vall += 2.0*v;
	}

	for (int i = 0; i < ksz; ++i) {
		kern[i] /= vall;
	}
}

template<class T>
void blurGrid(const Grid<T>& source, Grid<T>& target, Real kernelW, Real sigm = -0.1 ) {
	// adjust params properly
	int ksz = int(kernelW + 0.5);
	std::vector<Real> kern;
	get1DKernel(kern, ksz, sigm);

	// apply Conv
	Grid<T> tmpGrid(source);
	knConvGrid1D<T>(source, tmpGrid, kern, 0);//blur x
	knConvGrid1D<T>(tmpGrid, target, kern, 1);//blur y
	if (target.is3D()) {//blur z
		tmpGrid.copyFrom(target);
		knConvGrid1D<T>(tmpGrid, target, kern, 2);
	}
}

KERNEL() template<class S, class T>
void knSaveLocalPatchGrid(Grid<T>& localGrid, Grid<T>& tarGrid, Vec3i& desRegion, PatchSystem<S>& pp, int pidx,
	Grid<int>* preIndexGrid, Grid<Vec3>* preTetGrid) {
	
	Vec3 cubepos = Vec3(i + 0.5f, j + 0.5f, k + 0.5f) / toVec3(desRegion);
	Vec3 smokepos(0.0);
	
	int preID = -1; Vec3 uvw(0.0);
	if ( preIndexGrid ) {
		Vec3 pfactor = calcGridSizeFactor(preIndexGrid->getSize(), localGrid.getSize());
		if (pp.getParent()->is2D()) pfactor.z = 1.0;

		if ( localGrid.getSize() * toVec3i(pfactor + Vec3(0.5)) != preIndexGrid->getSize() )
			preTetGrid = NULL;
		
		Vec3i prepos = toVec3i(Vec3(i + 0.5f, j + 0.5f, k + 0.5f) * pfactor);
		if (preIndexGrid->isInBounds(prepos)) {
			preID = (*preIndexGrid)(prepos);
			if (preTetGrid)
				uvw = (*preTetGrid)(prepos);
		}
	}
	if ( preID < 0 )
		smokepos = pp.patlocal2World(pidx, cubepos, NULL);
	else if (preTetGrid == NULL) {
		smokepos = pp.patlocal2World(pidx, cubepos, &preID);
	}
	else // have both, most accelerated version
		smokepos = pp.getDefoPos(pidx, preID, uvw);

	smokepos = smokepos * calcGridSizeFactor(tarGrid.getSize(), pp.getParent()->getGridSize());
	if (pp.getParent()->is2D())
		smokepos.z = 0.5;
			
	if (tarGrid.isInBounds(smokepos))
		localGrid(i, j, k) = tarGrid.getInterpolatedHi(smokepos, 2);
	else
		localGrid(i, j, k) = T(0.0f);
}

template<class S, class T>
void saveLocalPatchGrid(int idx, Grid<T>& tarGrid, Grid<T>& localGrid, PatchSystem<S>& pp, Grid<int>* preIndexGrid, Grid<Vec3>* preTetGrid)
{
	bool is2D = !(tarGrid.is3D());
	//Vec3 sizeFactor = calcGridSizeFactor(tarGrid.getSize(), pp.getParent()->getGridSize());
	Vec3i desRegion = localGrid.getParent()->getGridSize();
	knSaveLocalPatchGrid<S, T>(localGrid, tarGrid, desRegion, pp, idx, preIndexGrid, preTetGrid);
}

KERNEL() template<class S, class T>
void knSaveLocalPatchGridNumpy(Grid<T>& localGrid, Grid<T>& tarGrid, Vec3i& desRegion, 
	PatchSystem<S>& pp, int pidx, PyArrayContainer realLocalGrid, PyArrayContainer bglist,
	Grid<int>* preIndexGrid, Grid<Vec3>* preTetGrid) {

	Vec3 cubepos = Vec3(i + 0.5f, j + 0.5f, k + 0.5f) / toVec3(desRegion);
	Vec3 smokepos(0.0);
	int preID = -1; Vec3 uvw(0.0);

	if (preIndexGrid) {
		Vec3 pfactor = calcGridSizeFactor(preIndexGrid->getSize(), localGrid.getSize());
		if (pp.getParent()->is2D()) pfactor.z = 1.0;

		if (localGrid.getSize() * toVec3i(pfactor + Vec3(0.5)) != preIndexGrid->getSize())
			preTetGrid = NULL;

		Vec3i prepos = toVec3i(Vec3(i + 0.5f, j + 0.5f, k + 0.5f) * pfactor);
		if (preIndexGrid->isInBounds(prepos)) {
			preID = (*preIndexGrid)(prepos);
			if (preTetGrid)
				uvw = (*preTetGrid)(prepos);
		}
	}
	if (preID < 0)
		smokepos = pp.patlocal2World(pidx, cubepos, NULL);
	else if (preTetGrid == NULL) {
		smokepos = pp.patlocal2World(pidx, cubepos, &preID);
	}
	else // have both, most accelerated version
		smokepos = pp.getDefoPos(pidx, preID, uvw);
	
	int idx2 = localGrid.index(i, j, k) + (reinterpret_cast<int*>(bglist.pData))[pidx] *
		(localGrid.getSizeX() * localGrid.getSizeY() *localGrid.getSizeZ());

	smokepos = smokepos * calcGridSizeFactor(tarGrid.getSize(), pp.getParent()->getGridSize());
	if (pp.getParent()->is2D())
		smokepos.z = 0.5;

	T* dataidx2 = NULL;
	realLocalGrid.get(idx2, dataidx2);
	if (dataidx2) {
		if (tarGrid.isInBounds(smokepos))
			(*dataidx2) = tarGrid.getInterpolatedHi(smokepos, 2);
		else
			(*dataidx2) = T(0.0f);
	}
}

KERNEL() template<class S>
void knSaveLocalPatchMACGridNumpy(MACGrid& localGrid, MACGrid& tarGrid, Vec3i& desRegion,
	PatchSystem<S>& pp, int pidx, PyArrayContainer realLocalGrid, PyArrayContainer bglist,
	Grid<int>* preIndexGrid, Grid<Vec3>* preTetGrid) {

	Vec3 cubepos = Vec3(i + 0.5f, j + 0.5f, k + 0.5f) / toVec3(desRegion);
	Vec3 smokepos(0.0);
	int preID = -1; Vec3 uvw(0.0);

	if (preIndexGrid) {
		Vec3 pfactor = calcGridSizeFactor(preIndexGrid->getSize(), localGrid.getSize());
		if (pp.getParent()->is2D()) pfactor.z = 1.0;

		if (localGrid.getSize() * toVec3i(pfactor + Vec3(0.5)) != preIndexGrid->getSize())
			preTetGrid = NULL;

		Vec3i prepos = toVec3i(Vec3(i + 0.5f, j + 0.5f, k + 0.5f) * pfactor);
		if (preIndexGrid->isInBounds(prepos)) {
			preID = (*preIndexGrid)(prepos);
			if (preTetGrid)
				uvw = (*preTetGrid)(prepos);
		}
	}
	if (preID < 0)
		smokepos = pp.patlocal2World(pidx, cubepos, NULL);
	else if (preTetGrid == NULL) {
		smokepos = pp.patlocal2World(pidx, cubepos, &preID);
	}
	else // have both, most accelerated version
		smokepos = pp.getDefoPos(pidx, preID, uvw);

	int idx2 = localGrid.index(i, j, k) + (reinterpret_cast<int*>(bglist.pData))[pidx] *
		(localGrid.getSizeX() * localGrid.getSizeY() *localGrid.getSizeZ());

	smokepos = smokepos * calcGridSizeFactor(tarGrid.getSize(), pp.getParent()->getGridSize());
	if (pp.getParent()->is2D())
		smokepos.z = 0.5;

	Vec3* dataidx2 = NULL;
	realLocalGrid.get(idx2, dataidx2);
	if (dataidx2) {
		if (tarGrid.isInBounds(smokepos))
			(*dataidx2) = tarGrid.getInterpolatedHi(smokepos, 2);
		else
			(*dataidx2) = Vec3(0.0f);
	}
}

KERNEL() template<class S>
void knSaveLocalPatchMACGrid(Grid<Vec3>& localGrid, MACGrid& tarGrid, Vec3i& desRegion, PatchSystem<S>& pp, int pidx,
	Grid<int>* preIndexGrid, Grid<Vec3>* preTetGrid) {

	Vec3 cubepos = Vec3(i + 0.5f, j + 0.5f, k + 0.5f) / toVec3(desRegion);
	Vec3 smokepos(0.0);

	int preID = -1; Vec3 uvw(0.0);
	if (preIndexGrid) {
		Vec3 pfactor = calcGridSizeFactor(preIndexGrid->getSize(), localGrid.getSize());
		if (pp.getParent()->is2D()) pfactor.z = 1.0;

		if (localGrid.getSize() * toVec3i(pfactor + Vec3(0.5)) != preIndexGrid->getSize())
			preTetGrid = NULL;

		Vec3i prepos = toVec3i(Vec3(i + 0.5f, j + 0.5f, k + 0.5f) * pfactor);
		if (preIndexGrid->isInBounds(prepos)) {
			preID = (*preIndexGrid)(prepos);
			if (preTetGrid)
				uvw = (*preTetGrid)(prepos);
		}
	}
	if (preID < 0)
		smokepos = pp.patlocal2World(pidx, cubepos, NULL);
	else if (preTetGrid == NULL) {
		smokepos = pp.patlocal2World(pidx, cubepos, &preID);
	}
	else // have both, most accelerated version
		smokepos = pp.getDefoPos(pidx, preID, uvw);

	smokepos = smokepos * calcGridSizeFactor(tarGrid.getSize(), pp.getParent()->getGridSize());
	if (pp.getParent()->is2D())
		smokepos.z = 0.5;

	if (tarGrid.isInBounds(smokepos))
		localGrid(i, j, k) = tarGrid.getInterpolatedHi(smokepos, 2);
	else
		localGrid(i, j, k) = Vec3(0.0f);
}

template<class S>
void saveLocalPatchMACGridTempl(int idx, MACGrid& tarGrid, Grid<Vec3>& localGrid, PatchSystem<S>& pp, Grid<int>* preIndexGrid, Grid<Vec3>* preTetGrid)
{
	bool is2D = !(tarGrid.is3D());
	//Vec3 sizeFactor = calcGridSizeFactor(tarGrid.getSize(), pp.getParent()->getGridSize());
	Vec3i desRegion = localGrid.getParent()->getGridSize();
	knSaveLocalPatchMACGrid<S>(localGrid, tarGrid, desRegion, pp, idx, preIndexGrid, preTetGrid);
}

}

#endif