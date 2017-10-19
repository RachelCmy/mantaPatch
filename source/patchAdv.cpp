#include "patchAdv.h"
#include "simpleimage.h"
#include "commonkernels.h"

#include <string>
#if defined _MSC_VER
#include <direct.h>
#elif defined __GNUC__
#include <sys/types.h>
#include <sys/stat.h>
#endif
void createDir(std::string dir) {
#if defined _MSC_VER
	_mkdir(dir.data());
#elif defined __GNUC__
// Rachel: haven't tested on other platform!
	mkdir(dir.data(), 0777); //0755 ?
#endif
}

namespace Manta {

extern void extrapolateMACSimple(FlagGrid& flags, MACGrid& vel, int distance = 4,
	LevelsetGrid* phiObs = NULL, bool intoObs = false);
extern void projectPpmFull(Grid<Real>& val, std::string name, int shadeMode, Real scale);
extern void quantizeGrid(Grid<Real>& grid, Real step);
extern void resampleVec3ToMac(Grid<Vec3>& source, MACGrid &target);
// !Weighting functions


void getGridBlockCentre(const Grid<Real>& sourceGrid, Grid<Real>& centreGrid, Real sidelen = 9.0) {

	Grid<Vec3> SATWGrid(sourceGrid.getParent(), false);
	FOR_IJK(SATWGrid) {
		SATWGrid(i, j, k) = Vec3(i + 0.5, j + 0.5, k + 0.5) * sourceGrid(i, j, k);
	}
	knGetGridSAT1D<Vec3>(SATWGrid, 0);
	knGetGridSAT1D<Vec3>(SATWGrid, 1);
	knGetGridSAT1D<Vec3>(SATWGrid, 2);

	Grid<Real> SATGrid(sourceGrid);
	SATGrid.copyFrom(sourceGrid);
	knGetGridSAT1D<Real>(SATGrid, 0);
	knGetGridSAT1D<Real>(SATGrid, 1);
	knGetGridSAT1D<Real>(SATGrid, 2);

	centreGrid.setConst(0.0f);
	bool is2D = !(sourceGrid.is3D());
	Vec3i scgz = sourceGrid.getSize();
	Vec3i seedsnum = toVec3i(toVec3(scgz) / sidelen + Vec3(0.5f, 0.5f, 0.5f));
	if (is2D) seedsnum.z = 1;

	int totalnum1 = seedsnum.x*seedsnum.y*seedsnum.z;
	seedsnum = seedsnum - Vec3i(1, 1, 1);// half block offset
	if (is2D) seedsnum.z = 1;
	int totalnum2 = seedsnum.x*seedsnum.y*seedsnum.z;
	
	#pragma omp for
	for (int celli = 0; celli < (totalnum1 + totalnum2); ++celli) {
		Vec3i bgpos(0, 0, 0);
		if (celli < totalnum1) {
			bgpos.x = sidelen * (celli % (seedsnum.x + 1));
			bgpos.y = sidelen * ((celli / (seedsnum.x + 1)) % (seedsnum.y + 1));
			if (!is2D) bgpos.z = sidelen * ((celli / (seedsnum.x + 1)) / (seedsnum.y + 1));
		}
		else {
			int newcelli = celli - totalnum1;
			bgpos.x = sidelen * (newcelli % seedsnum.x + 0.5f) + 0.5f;
			bgpos.y = sidelen * ((newcelli / seedsnum.x) % seedsnum.y + 0.5f) + 0.5f;
			if (!is2D) bgpos.z = sidelen * ((newcelli / seedsnum.x) / seedsnum.y + 0.5f) + 0.5f;
		}

		Vec3i edpos = bgpos + Vec3i(sidelen - 1, sidelen - 1, sidelen - 1);
		if (edpos.x >= scgz.x) edpos.x = scgz.x - 1;
		if (edpos.y >= scgz.y) edpos.y = scgz.y - 1;
		if (edpos.z >= scgz.z) edpos.z = scgz.z - 1;

		//knFindCellCentre<T>(sourceGrid, centreGrid, bgpos, sidelen, (is2D ? 1 : sidelen));

		Vec3 totalW = getSumFromSAT<Vec3>(SATWGrid, bgpos, edpos);
		Real totalX = getSumFromSAT<Real>(SATGrid, bgpos, edpos);

		if (totalX > VECTOR_EPSILON) {
			Vec3 newpos = totalW / totalX;
			if (newpos.x >= bgpos.x && newpos.y >= bgpos.y && newpos.x <= edpos.x && newpos.y <= edpos.y) {
				if (is2D || (newpos.z >= bgpos.z && newpos.z <= edpos.z)) {
					Vec3 sourceFactor = calcGridSizeFactor(centreGrid.getSize(), sourceGrid.getSize());
					Vec3i newposi = toVec3i(newpos * sourceFactor);
					if (is2D) newposi.z = 0;
					centreGrid(newposi) = totalX;
				}
			}
		}
	}
}

KERNEL() template<class S, class T>
void knAccuPatch(Grid<Real>& tarWeiG, PatchSystem<S>& pp, bool withDefo, bool withKern, bool withFad,// accumulate patch Wei to tarWeiG
	T* tarG = NULL, std::vector<T*>* listGrid = NULL,  // accumulate listGrid to tarG
	std::vector<int> * cellList = NULL, std::vector<int> * tetList = NULL, bool withFadingOut = true) {
	// per-cell acceleration data structure!
	// acceleration is good only if tarWeiG is larger than pp.getParent, in terms of resolution

	if (tarG && listGrid) {
		assertMsg(tarWeiG.getSize() == tarG->getSize(), "tarWeiG and tarG should have a same resolution.");
	}

	bool is2D = pp.getParent()->is2D();
	Vec3i gs = pp.getParent()->getGridSize();
	Vec3 factor = calcGridSizeFactor(tarWeiG.getSize(), gs);
	std::vector< int  > *plist = NULL, *tlist = NULL;
	int psize = 0;

	if (cellList != NULL) {// has acceleration, use the par list and tet list
		Vec3i curp = toVec3i(Vec3(i + 0.5, j + 0.5, k + 0.5) / factor);
		int index = (curp.z*gs.y + curp.y) *gs.x + curp.x;
		if (index < (gs.x*gs.y*gs.z)) {
			plist = &(cellList[index]);
			tlist = &(tetList[index]);
			psize = (*plist).size();
		}
	}
	else {// no acceleration, use full list
		psize = pp.size();
		plist = new std::vector<int>(psize);
		//tlist = new std::vector<int>(psize);
		for (int pid = 0; pid < psize; ++pid) {
			(*plist)[pid] = pid;
			//(*tlist)[pid] = -1; // unknow
		}
	}

	for (int pid = 0; pid < psize; ++pid) {
		const int idx = (*plist)[pid];
		//if (idx != 5) continue; // debug
		if (!pp.isActive(idx)) continue;
		if (withFadingOut == false && pp.isFadingOut(idx)) continue;
		Vec3 curp = Vec3(i + 0.5, j + 0.5, k + 0.5) / factor;
		Vec3 patchPos(curp);//patchPos [0-1]

		if (withDefo) {
			int * tetID = (cellList == NULL) ? NULL : &((*tlist)[pid]);
			bool inside = false;
			patchPos = pp.world2patLocal(idx, curp, &inside, tetID);
			if (inside == false) continue;
		}
		else {
			patchPos = (curp - pp.getPos(idx)) / (pp.patResoLen) + Vec3(0.5, 0.5, 0.5);
		}
		if ((patchPos.x<0.0) ||
			(patchPos.y<0.0) ||
			(patchPos.z<0.0) ||
			(patchPos.x>1.0) ||
			(patchPos.y>1.0) ||
			(patchPos.z>1.0)) continue;


		float spaceWei = getConeWeight(norm(patchPos - Vec3(0.5)) * Real(2.0));
		if (withKern == false)
			spaceWei = 1.0f;
		if (spaceWei < 1e-6f) continue;
		Real fadingW = (withFad ? pp.getFading(idx):1.0f);
		float allwei = spaceWei * fadingW;
		if (allwei < 1e-6f) continue;
		
		// add patch value
		if (tarG != NULL && listGrid != NULL) {
			if ((*listGrid)[idx] != NULL) {
				//patchPos [0-1] -> (*listGrid)[iter]->getSize()
				patchPos = patchPos * toVec3((*listGrid)[idx]->getSize());
				if (is2D) patchPos.z = 0.5f;
				if (!(*listGrid)[idx]->isInBounds(patchPos)) continue;

				(*tarG)(i, j, k) += allwei * (*listGrid)[idx]->getInterpolatedHi(patchPos, 2);
				tarWeiG(i, j, k) += allwei;
			}
		} else
			tarWeiG(i, j, k) += allwei;
	}

	if (cellList == NULL) {
		delete plist;
		//delete tlist;
	}
}

template<class S>
void PatchSystem<S>::sampleCandidates(const Grid<Real>& denG, Real samWidth, Real weiThresh /*= 30.0*/,
	Real occThresh /*= 0.5*/, Grid<Real>* patchWeiGp /*= NULL*/){
	candpos_list.clear();
	Grid<Real>* patchWeiG = patchWeiGp;
	if (patchWeiGp == NULL) {// generate patchWeiG without acceleration
		patchWeiG = new Grid<Real>(denG.getParent(), false);
		knAccuPatch<S, Grid<Real>>(*patchWeiG, *this, false, true, false, NULL, NULL, NULL, NULL, false);
	}

	Grid<Real> centreG(denG.getParent(), false);
	getGridBlockCentre(denG, centreG, samWidth);

	Vec3 sourceFactorOW = calcGridSizeFactor(patchWeiG->getSize(), centreG.getSize());
	Vec3 sourceFactorPW = calcGridSizeFactor(getParent()->getGridSize(), centreG.getSize());
	bool is2D = getParent()->is2D();
	if (is2D) sourceFactorOW.z = sourceFactorPW.z = 1.0f;

	static RandomStream randStreamTile(1233906);// save as member data??
	Real jitR = samWidth / 4.0;
	int counter = candpos_list.size();
	int canID = 0;
	FOR_IJK(centreG) {// cannot parallel!
		Vec3i posi(i, j, k); Vec3  posf(i + 0.5f, j + 0.5f, k + 0.5f);
		Real wV = centreG.get(posi);
		if (wV > weiThresh) {
			Real oV = patchWeiG->get(toVec3i(posf * sourceFactorOW)), bV = denG.get(posi);
			if (oV < occThresh) { // sample with jitter!
				if (jitN <= 1) {
					candpos_list.push_back(S(posf * sourceFactorPW));
					candpos_list[counter++].uniqID = (canID);
				}
				else{
					for (int s = 0; s < jitN; ++s) {
						int maxi = 0;
						Vec3 posfr = posf;
						do {// try a valid random position as candidates
							if (maxi > 5) {//
								posfr = posf;
								break; // fail
							}
							Vec3 offset = randStreamTile.getVec3Norm();
							if (is2D) offset.z = 0.0f;
							posfr = (posf * sourceFactorPW + offset * jitR) / sourceFactorPW;
							oV = patchWeiG->get(toVec3i(posfr * sourceFactorOW));
							bV = denG.get(toVec3i(posfr));
							++maxi;
						} while (oV > occThresh || bV < VECTOR_EPSILON);
						candpos_list.push_back(S(posfr * sourceFactorPW));
						candpos_list[counter++].uniqID = (canID);
					}
				}//	++counter;
				++canID;
			}
		}
	}
	if (patchWeiGp == NULL && patchWeiG != NULL) {
		delete patchWeiG;
	}
}

// length will be the length of curdir,only to reduce one calculation...
int getBinforDir(Vec3 curdir, bool is3D, int binN, Real * length = NULL) {

	if(length) 
		*length = normalize(curdir);
	else 
		normalize(curdir);

	float theta1/*z-xy*/, theta2/*y-x, 0-2PI*/;
	if (fabs(curdir.x) <= 1e-6f)
		theta2 = (curdir.y > 0.0f) ? M_PI / 2.0f : (3.0f * M_PI / 2.0f);
	else {
		theta2 = atan(curdir.y / curdir.x);//assume 2D, z = 0
		if (curdir.x < 0.0f) theta2 += M_PI;
		else if (curdir.y < 0.0f) theta2 += 2.0f * M_PI;
	}
	int binidx2 = (int)( theta2 / (2.0 * M_PI) * binN );
	if (binidx2 >= binN) binidx2 -= binN;

	int binidx1 = 0;
	if (is3D) {
		
		if (curdir.z >= 1.0 - VECTOR_EPSILON) binidx1 = 0;
		else if (curdir.z <= -1.0 + VECTOR_EPSILON) binidx1 = binN / 2 - 1;
		else {
			theta1 = acos(curdir.z);

			binidx1 = (int)(theta1 / M_PI * (binN / 2));
			if (binidx1 >= binN / 2) binidx1 = binN / 2 - 1;
		}
	}
	return (binidx1 * binN + binidx2);
}

void dirHistAvg(std::vector<Real>& hist, bool is3D, int binN) {
	for (int id3 = 0; id3 < binN / 2; ++id3) {		
		for (int sn = 0; sn < 2; ++sn){
			double firstE = hist[id3 * binN];
			double last = hist[id3 * binN + binN - 1];
			for (int sw = 0; sw < binN; ++sw)
			{
				double cur = hist[id3 * binN + sw];
				double next = (sw == (binN - 1)) ? firstE : hist[id3 * binN + (sw + 1) % binN];
				hist[id3 * binN + sw] = (last + cur + next) / 3.0;
				last = cur;
			}
		}
		if (!is3D) return;
	}
	for (int id3 = 0; id3 < binN; ++id3)
	for (int sn = 0; sn < 2; ++sn) {
		hist[id3] = (hist[id3] + hist[id3 + binN]) / 2.0;
		hist[id3 + binN*(binN / 2 - 1)]	= (hist[id3 + binN*(binN / 2 - 1)] + hist[id3 + binN*(binN / 2 - 2)]) / 2.0;
		double last = hist[id3];
		for (int sw = 1; sw < binN / 2 - 1; ++sw) {
			double cur = hist[id3 + binN*sw];
			double next = hist[id3 + binN*(sw + 1)];
			hist[id3 + binN*sw] = (last + cur + next) / 3.0;
			last = cur;
		}
	}
}

// Fit a parabol to the three points (-1.0 ; left), (0.0 ; middle) and  
// (1.0 ; right).  
// Formulas:  
// f(x) = a (x - c)^2 + b, a should be negative so there is a maximum.
// c is the peak offset (where f'(x) is zero), b is the peak value.  
// In case there is an error false is returned, otherwise a correction  
// value between [-1 ; 1] is returned in 'degreeCorrection', where -1  
// means the peak is located completely at the left vector, and -0.5 just  
// in the middle between left and middle and > 0 to the right side. In  
// 'peakValue' the maximum estimated peak value is stored.  
bool InterpolateOrientation(double left, double middle, double right,
	double& degreeCorrection, double& peakValue)
{
	double a = ((left + right) - 2.0 * middle) / 2.0;
	if (a >= 0.0)//impossible! No max!
		return false;
	double c = (((left - middle) / a) - 1.0) / 2.0;
	if (c < -1.0 || c > 1.0)//too far away
		return false;
	double b = middle - c * c * a;

	degreeCorrection = c;
	peakValue = b;
	return true;
}


// !Functions for direction calculation
// origin SIFT method, quadratically interpolate the true peak
Vec3 getDirofHistogram(std::vector<Real>& hist, bool is3D, int binN, int& maxBin) {

	maxBin = std::distance(hist.begin(), std::max_element(hist.begin(), hist.end()));

	if (hist[maxBin] < VECTOR_EPSILON) return Vec3(0.0);

	Vec3 sumdir;
	double maxDegreeCorrection = 0.0;
	double maxDegreeCorrection2 = 0.0;
	double maxPeakValue = 0.0;
	double maxvalue = hist[maxBin];
	{
		int leftbin = maxBin - 1;
		if (maxBin % binN == 0) leftbin += binN;
		double leftvalue = hist[leftbin];
		if (leftvalue >= maxvalue) leftvalue = maxvalue;
		int rightbin = maxBin + 1;
		if (rightbin % binN == 0) rightbin -= binN;
		double rightvalue = hist[rightbin];
		if (rightvalue >= maxvalue) rightvalue = maxvalue;

		if (fabs(leftvalue - rightvalue) > 1e-6f) {
			InterpolateOrientation(leftvalue, maxvalue, rightvalue,
				maxDegreeCorrection, maxPeakValue); // false means symmetry!
		}
	}

	double maxxyId = maxBin%binN + 0.5 + maxDegreeCorrection;
	double maxdegreexy = maxxyId * 2.0 * M_PI / binN;

	if (!is3D) sumdir = Vec3(cos(maxdegreexy), sin(maxdegreexy), 0.0f);// already normalized
	else {
		{
			bool moved = false;
			int upbin = maxBin - binN;
			int downbin = maxBin + binN;
			float upvalue, downvalue;
			if (upbin < 0) {
				upbin = maxBin;
				maxBin = downbin;
				downbin = maxBin + binN;
				upvalue = hist[upbin];
				downvalue = hist[downbin];
				moved = true;
			}else if (downbin >= binN*binN / 2) {
				downbin = maxBin;
				maxBin = upbin;
				upbin = maxBin - binN;
				upvalue = hist[upbin];
				downvalue = hist[downbin];
				moved = true;
			}else {
				upvalue = hist[upbin];
				downvalue = hist[downbin];
				if (upvalue >= maxvalue) upvalue = maxvalue;
				if (downvalue >= maxvalue) downvalue = maxvalue;
			}

			maxPeakValue = 0.0;
			if (fabs(upvalue - downvalue) > 1e-6f) {
				if (!InterpolateOrientation(upvalue, maxvalue, downvalue, maxDegreeCorrection2, maxPeakValue)) {
					if (moved)
						maxBin = (upbin < binN) ? upbin : downbin;
				}
			}
		}
		double maxdegreez = (maxBin / binN + maxDegreeCorrection2 + 0.5) * 2.0 * M_PI / binN;
		sumdir = Vec3(cos(maxdegreexy) * sin(maxdegreez), sin(maxdegreexy)* sin(maxdegreez), cos(maxdegreez)); //already normalized
	}
	
	return sumdir;
}

template<class S>
void PatchSystem<S>::initCandidateCage(Grid<Real>& denG) {

	Grid<Real> blurG(denG.getParent(), false);
	blurGrid<Real>(denG, blurG, patResoLen);
	Grid<Vec3> graG(denG.getParent(), false);
	GradientOp(graG, blurG);	

	const int subd = 16; // for pi
	int samp = 9;
	bool is3D = denG.getParent()->is3D();
	Real step(patResoLen / Real(samp));
	Vec3 factor = calcGridSizeFactor(denG.getSize(), getParent()->getGridSize());
	std::vector<Real> Gkern;
	get1DKernel(Gkern, samp);

	int canN = candpos_list.size();
	#pragma omp for
	for (int idx = 0; idx < (canN); ++idx) {
		if ( (candpos_list[idx].flag & PDELETE) != 0 ) continue;
		if (candpos_list[idx].defoData.hasDefo()) continue;

		// new candidate
		std::vector<Real> orienHist(subd*2, 0.0);// for 2D
		if(is3D) orienHist.resize(subd * subd * 2, 0.0);// for 3D

		Vec3i samIDX(0);
		for (samIDX.z = 0; samIDX.z < (is3D ? samp : 1); samIDX.z +=1)
		for (samIDX.y = 0; samIDX.y < samp; samIDX.y += 1)
		for (samIDX.x = 0; samIDX.x < samp; samIDX.x += 1) {
			Vec3 samPos = candpos_list[idx].pos + Vec3(-patResoLen / 2.0, -patResoLen / 2.0, -patResoLen / 2.0) + ( toVec3(samIDX) + Vec3(0.5)) * step ;
			Vec3 graPos = samPos * factor;
			if (!is3D) graPos.z = 0.5;

			Vec3 curDir = graG.getInterpolated(graPos);
			Real curWei;
			int curBin = getBinforDir(curDir, is3D, subd * 2, &curWei);
			curWei *= Gkern[samIDX.x];
			curWei *= Gkern[samIDX.y];
			if(is3D) curWei *= Gkern[samIDX.z];

			orienHist[curBin] += curWei;
		}

		if (is3D) {// normalize his gram according to solid angle
			Real deltaTheta = (M_PI) / subd;
			for (size_t i = 0; i < orienHist.size(); ++i) {
				Real currenTheta2 = (i / (2*subd)) * deltaTheta;
				Real solidA = cos(currenTheta2) - cos(currenTheta2 + deltaTheta); // always positive
				orienHist[i] /= solidA;// * deltaPhi
			}
		}
		dirHistAvg(orienHist, is3D, subd * 2);
		int maxB = -1;
		Vec3 dirY = getDirofHistogram(orienHist, is3D, subd * 2, maxB); // already normalized!
		if (orienHist[maxB] < 1e-4) {
			candpos_list[idx].flag |= PDELETE;
			continue;
		}

		Vec3 dirX = cross(dirY, Vec3(0.0, 0.0, 1.0));
		// normalize!
		normalize(dirX);
		if (is3D) {
			if (norm(dirX) < VECTOR_EPSILON) dirX = Vec3(0.0, 1.0, 0.0);

			orienHist.resize(0);
			orienHist.resize(subd * 2, 0.0);
			Vec3 dirOff1(0.0), dirOff2(0.0);
			bool once = true;
			for (samIDX.z = 0; samIDX.z < samp; samIDX.z +=1)
			for (samIDX.y = 0; samIDX.y < samp; samIDX.y += 1)
			for (samIDX.x = 0; samIDX.x < samp; samIDX.x += 1) {
				Vec3 samPos = candpos_list[idx].pos + Vec3(-patResoLen / 2.0, -patResoLen / 2.0, -patResoLen / 2.0) + ( toVec3(samIDX) + Vec3(0.5)) * step ;
				Vec3 graPos = samPos * factor;

				Vec3 curDir = graG.getInterpolated(graPos);
				curDir -= dirY * dot(curDir, dirY); // project to orthogonal plane 
				// project to 2D!
				Real curWei = normalize(curDir);
				if (curWei < VECTOR_EPSILON) continue;
				if (once) {
					dirOff1 = curDir;
					dirOff2 = cross(dirOff1, dirY);// normalized already
					once = false;
				}

				int curBin = getBinforDir( Vec3( dot(curDir, dirOff2), dot(curDir, dirOff1) , 0.0), false, subd * 2);
				curWei *= Gkern[samIDX.x];
				curWei *= Gkern[samIDX.y];
				curWei *= Gkern[samIDX.z];

				orienHist[curBin] += curWei;
			}
			if (!once) {
				dirHistAvg(orienHist, false, subd * 2);
				dirX = getDirofHistogram(orienHist, false, subd * 2, maxB); // already normalized!
				dirX = dirOff2*dirX.x + dirOff1 * dirX.y; // still normalized!
			}
		}

		candpos_list[idx].initDefoPos(dirX, dirY, patResoLen, patRest);
	}

}

template<class S>
void PatchSystem<S>::addCandidatePatch() {
	int canN = candpos_list.size();
	for (int idx = 0; idx < (canN); ++idx) {
		if ((candpos_list[idx].flag & PDELETE) != 0) continue;
		if (!candpos_list[idx].defoData.hasDefo()) continue;

		candpos_list[idx].flag |= PNEW;
		candpos_list[idx].uniqID += pCount;
		IndexInt resultID = ParticleSystem<S>::add(candpos_list[idx]);
	}
	canN = size();
	pCount = mData[canN - 1].uniqID + 1;
	candpos_list.resize(0);
	//debMsg(pCount, 0);
}

KERNEL(pts) template<class S>
void knNeibParSysInit(PatchSystem<S>& pp, BasicParticleSystem& pNb, const int vn) {
	bool is2D = pp.getParent()->is2D();

	if (pp.isActive(idx)) {
		for (int vi = 0; vi < vn; ++vi) {
			Vec3 cornerPos = pp.getDefoCornerPos(idx, vi);
			if (is2D) cornerPos.z = 0.5f; // [-0.5,0.5]
			pNb.setPos(idx * vn + vi, cornerPos);
		}
	} else {
		for (int vi = 0; vi < vn; ++vi) {
			pNb.setPos(idx * vn + vi, Vec3(0.0));
			pNb.kill(idx * vn + vi);
		}
	}
}

KERNEL(pts) template<class S>
void knNeibParSysSetPDD(PatchSystem<S>& pp, BasicParticleSystem& pNb, const int vn, const int subd, const Real lambda, const bool scaleLen) {
	bool is2D = pp.getParent()->is2D();
	if (pp.isActive(idx)) {
		std::vector<Vec3> points(vn);
		if (is2D) points.resize(vn * (2 + subd));

		for (int vi = 0; vi < vn; ++vi) {
			Vec3 curpos = pNb.getPos(idx * vn + vi);// the advected position
			if (is2D) {
				for (int k = 0; k < (2 + subd); ++k) {
					curpos.z = pp.getRestPDDPos(vi + vn * k).z * pp.patResoLen;//reset z 
					points[vi + vn * k] = curpos;
				}
			}
			else
				points[vi] = curpos;
		}

		pp.adjustDefo(idx, points, lambda);
		pp.setCentrePos(idx);
		if (scaleLen) {
			//pp.scaleCage(idx); // simple scale, has bugs!!
			pp.adjustScale(idx); // controlling length of the edges
		}
		pp.resetDefoMatrix(idx);
	}
}

template<class S>
void PatchSystem<S>::getCagePosParSys(BasicParticleSystem& target) {
	bool is2D = getParent()->is2D();
	int vn = patRest.restPoints.size();

	if (is2D) vn /= (2 + patRest.subdN);
	target.resizeAll(vn * size());
	knNeibParSysInit<S>(*this, target, vn);
}

template<class S>
void PatchSystem<S>::AdvectWithControl(const Real lamda, FlagGrid& flags, const MACGrid& vel, const int integrationMode, bool scaleLen) {

	MACGrid blurG(vel.getParent(), false);
	blurGrid<Vec3>(vel, blurG, patResoLen / Real(1.0 + patRest.subdN));

	extrapolateMACSimple(flags, blurG, patResoLen / 2.0, NULL, true);

	pNeib = new BasicParticleSystem(getParent());
	getCagePosParSys(*pNeib);
	pNeib->advectInGrid(flags, blurG, integrationMode, false, false);

	int vn = patRest.restPoints.size();
	if (getParent()->is2D()) vn /= (2 + patRest.subdN);
	knNeibParSysSetPDD<S>(*this, *pNeib, vn, patRest.subdN, lamda, scaleLen);

	delete pNeib;
	pNeib = NULL;
}

template<class S>
void PatchSystem<S>::synPerCellAcceleration(Vec3i tarSZ) {
	// need to rebuild according to new patch distribution & patch defodata
	if (cellList) delete[] cellList;
	if (tetList) delete[] tetList;

	Vec3i gs = getParent()->getGridSize();
	if (tarSZ.x < 1) tarSZ = gs;

	const Vec3 delta = calcGridSizeFactor(gs, tarSZ);

	const int dirNum = 1 + 6 + 8; // how many points to check
	Vec3 dpos[1 + 6 + 8] = {
		Vec3(0,0,0),
		Vec3(1. * delta[0], 0. * delta[1], 0. * delta[2]),
		Vec3(-1. * delta[0], 0. * delta[1], 0. * delta[2]),
		Vec3(0. * delta[0], 1. * delta[1], 0. * delta[2]),
		Vec3(0. * delta[0],-1. * delta[1], 0. * delta[2]),
		Vec3(0. * delta[0], 0. * delta[1], 1. * delta[2]),
		Vec3(0. * delta[0], 0. * delta[1],-1. * delta[2]) , // 7

		Vec3(1. * delta[0], 1. * delta[1], 1. * delta[2]),
		Vec3(-1. * delta[0], 1. * delta[1], 1. * delta[2]),
		Vec3(1. * delta[0],-1. * delta[1], 1. * delta[2]),
		Vec3(-1. * delta[0],-1. * delta[1], 1. * delta[2]),
		Vec3(1. * delta[0], 1. * delta[1],-1. * delta[2]),
		Vec3(-1. * delta[0], 1. * delta[1],-1. * delta[2]),
		Vec3(1. * delta[0],-1. * delta[1],-1. * delta[2]),
		Vec3(-1. * delta[0],-1. * delta[1],-1. * delta[2])
	};
	cellList = new std::vector<int> [gs.x*gs.y*gs.z];
	tetList = new std::vector<int>[gs.x*gs.y*gs.z];
	for (int z = 0; z < gs.z; z++)
	for (int y = 0; y < gs.y; y++)
	for (int x = 0; x < gs.x; x++) {
		const int index = (z*gs.y + y) *gs.x + x;

		// cell
		Vec3 pw = Vec3(x + 0.5f, y + 0.5f, z + 0.5f);

		FOR_PARTS(*this) {
			if (!isActive(idx)) continue;
			if (!mData[idx].defoData.inbbCheck(pw)) continue;

			float fdist = 1.05;
			int tetidx = -1;
			Vec3 uvw;
			for (int jj = 0; jj < dirNum; ++jj) {
				Vec3 ppj = (pw + fdist * dpos[jj]);
				tetidx = mData[idx].defoData.isInside(ppj, &uvw);
				if (tetidx >= 0) {
					cellList[index].push_back(idx);
					tetList[index].push_back(tetidx/4);
					break;
				}
			}
		}
	}
}

template<class S>
void PatchSystem<S>::patchSynthesisReal(Grid<Real>& tarG, Vec3i patchSZ, PyArrayContainer patchGrids,
	PyArrayContainer* patchDict, bool withSpaceW, Grid<Real>* weigG, bool clear) {
	int dim = tarG.is3D() ? 3 : 2;
	FluidSolver patchSlv(patchSZ, dim);
	//patchGrids, shape [patchN, patchSZ.x, patchSZ.y, patchSZ.z]
	int patsz = patchSZ.x*patchSZ.y*patchSZ.z;
	int patchN = patchGrids.TotalSize / (patsz);
	Grid<Real>** gdList = new Grid<Real>*[patchN];

	for (int gdi = 0; gdi < patchN; ++gdi) {
		gdList[gdi] = new Grid<Real>(&patchSlv, false);
		FOR_IDX(*gdList[gdi]) {
			(*gdList[gdi])(idx) = reinterpret_cast<float*>(patchGrids.pData)[idx + gdi * patsz];
		}
	}
	// rearrange according to dictionary
	int curSz = size();
	std::vector<Grid<Real>*> patList(curSz, NULL);
	// patchDict, shape [patchN]
	if (patchDict) {
		const int* dic = reinterpret_cast<int*>(patchDict->pData);
		for (int pci = 0; pci < curSz; ++pci) {
			if (dic[pci] < 0) 
				patList[pci] = NULL;
			else
				patList[pci] = gdList[ dic[pci] ];
		}
	} else {
		for (int pci = 0; pci < curSz; ++pci) {
			patList[pci] = gdList[pci % patchN];
		}
	}
	Grid<Real> weiG(tarG.getParent(), false);
	if (clear) {
		weiG.setConst(0.0);
		tarG.setConst(0.0);
	} else {
		if (weigG) {
			if (weigG->getSize() == weiG.getSize()) {
				weiG.copyFrom(*weigG);
			}
			else {
				Vec3 sourceFactor = calcGridSizeFactor(weigG->getSize(), weiG.getSize());
				knInterpolateGridTempl<Real>(weiG, *weigG, sourceFactor, Vec3(0.0), 2);
			}
		}
	}
	knAccuPatch<S, Grid<Real>>(weiG, *this, true, withSpaceW, true, &tarG, &patList, cellList, tetList);

	if (weigG) {
		if (weigG->getSize() == weiG.getSize()) {
			weigG->copyFrom(weiG);
		}
		else {
			Vec3 sourceFactor = calcGridSizeFactor(weiG.getSize(), weigG->getSize());
			knInterpolateGridTempl<Real>(*weigG, weiG, sourceFactor, Vec3(0.0), 2);
		}
	}
	//weiG.clamp(1.0, 100000.0);
	//tarG.safeDivide(weiG);
	
	for (int gdi = 0; gdi < patchN; ++gdi)
		delete gdList[gdi];
	delete[] gdList;
}

KERNEL() template<class S>
void knUndeformV(Grid<Vec3>& localGrid, PatchSystem<S>& pp, int pidx, Grid<int>* preIndexGrid) {
	Vec3i desRegion = localGrid.getSize();
	Vec3 cubepos = Vec3(i + 0.5f, j + 0.5f, k + 0.5f) / toVec3(desRegion);
	int preID = -1;
	if (preIndexGrid) {
		Vec3 pfactor = calcGridSizeFactor(preIndexGrid->getSize(), localGrid.getSize());
		Vec3i prepos = toVec3i(Vec3(i + 0.5f, j + 0.5f, k + 0.5f) * pfactor);

		if (pp.getParent()->is2D()) prepos.z = 0;
		if (preIndexGrid->isInBounds(prepos)) {
			preID = (*preIndexGrid)(prepos);
		}
	}
	Vec3 undV = pp.worldVec2patLocalVec(pidx, cubepos, localGrid(i, j, k), &preID);
	localGrid(i, j, k) = undV;
}

KERNEL() template<class S>
void knUndeformVNumpy(Grid<Vec3>& localGrid, PyArrayContainer patchGrids, PyArrayContainer patchDict,
	PatchSystem<S>& pp, int pidx, Grid<int>* preIndexGrid) {
	if ((reinterpret_cast<int*>(patchDict.pData))[pidx] < 0) return;
	Vec3i desRegion = localGrid.getSize();
	Vec3 cubepos = Vec3(i + 0.5f, j + 0.5f, k + 0.5f) / toVec3(desRegion);
	int preID = -1;
	if (preIndexGrid) {
		Vec3 pfactor = calcGridSizeFactor(preIndexGrid->getSize(), localGrid.getSize());
		Vec3i prepos = toVec3i(Vec3(i + 0.5f, j + 0.5f, k + 0.5f) * pfactor);

		if (pp.getParent()->is2D()) prepos.z = 0;
		if (preIndexGrid->isInBounds(prepos)) {
			preID = (*preIndexGrid)(prepos);
		}
	}

	int idx2 = localGrid.index(i, j, k) + (reinterpret_cast<int*>(patchDict.pData))[pidx] *
		(desRegion.x * desRegion.y *desRegion.z);

	Vec3 oldV = (reinterpret_cast<Vec3*>(patchGrids.pData))[idx2];
	Vec3 undV = pp.worldVec2patLocalVec(pidx, cubepos, oldV, &preID);
	(reinterpret_cast<Vec3*>(patchGrids.pData))[idx2] = undV;
}

KERNEL() template<class S>
void knDeformV(Grid<Vec3>& localGrid, PatchSystem<S>& pp, int pidx, Grid<int>* preIndexGrid) {
	Vec3i desRegion = localGrid.getSize();
	Vec3 cubepos = Vec3(i + 0.5f, j + 0.5f, k + 0.5f) / toVec3(desRegion);
	int preID = -1;
	if (preIndexGrid) {
		Vec3 pfactor = calcGridSizeFactor(preIndexGrid->getSize(), localGrid.getSize());
		Vec3i prepos = toVec3i(Vec3(i + 0.5f, j + 0.5f, k + 0.5f) * pfactor);

		if (pp.getParent()->is2D()) prepos.z = 0;
		if (preIndexGrid->isInBounds(prepos)) {
			preID = (*preIndexGrid)(prepos);
		}
	}
	Vec3 defV = pp.patlocalVec2WorldVec(pidx, cubepos, localGrid(i, j, k), &preID);
	localGrid(i, j, k) = defV;
}

template<class S>
void PatchSystem<S>::patchSynthesisMAC(MACGrid& tarG, Vec3i patchSZ, PyArrayContainer patchGrids,
	PyArrayContainer* patchDict, bool withSpaceW, Grid<Real>* weigG) {
	int dim = tarG.is3D() ? 3 : 2;
	FluidSolver patchSlv(patchSZ, dim);
	//patchGrids, shape [patchN, patchSZ.x, patchSZ.y, patchSZ.z, 3]
	int patsz = patchSZ.x*patchSZ.y*patchSZ.z;
	int patchN = patchGrids.TotalSize / (patsz * 3);
	int curSz = size();
	std::vector<Grid<Vec3>*> patList(curSz, NULL);
	
	for (int pci = 0; pci < curSz; ++pci) {
		if (!isActive(pci)) continue;
		int arrayID = pci % patchN;
		if (patchDict)
			arrayID = reinterpret_cast<int*>(patchDict->pData)[pci];
		if (arrayID < 0) continue;
		patList[pci] = new Grid<Vec3>(&patchSlv, false);
		FOR_IDX(*patList[pci]) {
			for (int vin = 0; vin < 3; vin++)
				(*patList[pci])(idx)[vin] = reinterpret_cast<float*>(patchGrids.pData)[3 * (idx + arrayID * patsz) + vin];
		}
	}

	// undeformed!!
	#pragma omp for // use another KERNEL(pts) leads to endless compiling
	for (int pci = 0; pci < curSz; ++pci) {
		if (patList[pci]) //  that's why there is no reuse as in patchSynthesisReal
			knDeformV<S>(*patList[pci], *this, pci, preIndexGrid);//use acceleration here!
	}

	Grid<Real> weiG(tarG.getParent(), false);
	weiG.setConst(0.0);
	tarG.setConst(Vec3(0.0));
	Grid<Vec3> tarGT(tarG.getParent(), false);
	knAccuPatch<S, Grid<Vec3> >(weiG, *this, true, withSpaceW, true, &tarGT, &patList, cellList, tetList);
	resampleVec3ToMac(tarGT, tarG);

	if (weigG) {
		if (weigG->getSize() == weiG.getSize()) {
			weigG->copyFrom(weiG);
		}
		else {
			Vec3 sourceFactor = calcGridSizeFactor(weiG.getSize(), weigG->getSize());
			knInterpolateGridTempl<Real>(*weigG, weiG, sourceFactor, Vec3(0.0), 2);
		}
	}
	
	for (int gdi = 0; gdi < patList.size(); ++gdi)
		if(patList[gdi]) delete patList[gdi];
}

void preComputeTetPos(Grid<int>& indexG, Grid<Vec3>& posG, std::vector<Vec3>& restTets, std::vector<Mat4>& defoInvRest) {

	Vec3i desRegion = indexG.getParent()->getGridSize();
	bool is2D = indexG.getParent()->is2D();
	FOR_IJK(indexG) {
		Vec3 cubepos = Vec3(i + 0.5f, j + 0.5f, k + 0.5f) / toVec3(desRegion);// [0,1]
		if (is2D) cubepos.z = 0.5f;
		Vec3 uvw;
		int aidx = isInsideOpt(restTets, defoInvRest, cubepos, Vec3(0.), Vec3(1.), &uvw);

		indexG(i, j, k) = aidx / 4;
		posG(i, j, k) = uvw;
	}
}

template<class S>
void PatchSystem<S>::saveLocalPerCellAcceleration(Vec3i accSZ)
{
	bool is2D = getParent()->is2D();
	if (preIndexGrid) delete preIndexGrid;
	if (preTetGrid) delete preTetGrid;

	if (accSZ.x < 1) accSZ = toVec3i(Vec3(patResoLen));
	if (is2D) accSZ.z = 1;

	if (preSolver && preSolver->getGridSize() != accSZ) {
		delete preSolver;
		preSolver = NULL;
	}
	if (preSolver == NULL)
		preSolver = new FluidSolver(accSZ, is2D ? 2 : 3);

	preIndexGrid = new Grid<int>(preSolver, false);
	preTetGrid = new  Grid<Vec3>(preSolver, false);

	preComputeTetPos(*preIndexGrid, *preTetGrid, patRest.restTets, patRest.defoInvRest);
}

void getPath(int uniqID, int frameID, bool createdir, std::string& dirpath, std::string& unipath, std::string& patchname) {
	int dirID = uniqID / 50;
	char tmpNum[4];
	sprintf_s(tmpNum, "%02d", uniqID % 50); tmpNum[2] = 0;
	patchname = std::string("P") + std::string(tmpNum);
	sprintf_s(tmpNum, "%03d", frameID); tmpNum[3] = 0;
	patchname = patchname + std::string("_") + std::string(tmpNum);

	sprintf_s(tmpNum, "%02d", dirID); tmpNum[2] = '/'; tmpNum[3] = 0;
	unipath = dirpath + std::string("/P") + std::string(tmpNum);
	if(createdir)
		createDir(unipath);
}

template<class S>
void PatchSystem<S>::saveLocalPatchGridReal(Grid<Real>& dataG, std::string dirPath, bool rescale, Real * step, bool ppmFlag, bool doFadOut)
{ // acceleration is recommended!
	bool is2D = (getParent()->is2D());
	Vec3 sizeFactor = calcGridSizeFactor(dataG.getSize(), getParent()->getGridSize());
	Vec3i desRegion = toVec3i(Vec3(patResoLen) * sizeFactor);
	if (is2D) desRegion.z = 1;
	FluidSolver tmpSolver(desRegion, is2D ? 2 : 3);
	Grid<Real> regionG(&tmpSolver, false); // reuse one grid for efficiency

	// no need to parallel, due to I/O access
	FOR_PARTS(*this) {// IO function, parallel won't be good
		if (!isActive(idx)) continue;
		if (!doFadOut && isFadingOut(idx)) continue;

		saveLocalPatchGrid<S, Real>(idx, dataG, regionG, *this, preIndexGrid, preTetGrid);
		if (rescale) {
			// ?
			Real minV = regionG.getMin();
			Real maxV = regionG.getMax();
			regionG.addConst(-minV);
			regionG.multConst( (1.0) / (maxV - minV));
		}

		if (step)// quantize! 
			quantizeGrid(regionG, *step);
		std::string unistr, patchname;
		getPath(mData[idx].uniqID, mData[idx].lifet, true, dirPath, unistr, patchname);
		
		if (ppmFlag)
			projectPpmFull(regionG, unistr + patchname + std::string(".ppm"), 0, 1.0);
		regionG.save(unistr + patchname + std::string(".uni"));
	}
}


KERNEL(idx) void knQuantizeV(Grid<Vec3>& grid, Real step)
{
	Vec3i    q = toVec3i(grid(idx) / step + /*step**/Vec3(0.5));
	Vec3 qd( q.x * (double)step, q.y * (double)step, q.z * (double)step);
	grid(idx) = qd;
}

PYTHON() void quantizeGridV(Grid<Vec3>& grid, Real step) { knQuantizeV(grid, step); }

template<class S>
void PatchSystem<S>::saveLocalPatchMACGrid(MACGrid & dataG, std::string dirPath, bool rescale, Real * step, bool ppmFlag, bool doFadOut)
{// acceleration is STRONGLY recommended!
	bool is2D = (getParent()->is2D());
	Vec3 sizeFactor = calcGridSizeFactor(dataG.getSize(), getParent()->getGridSize());
	Vec3i desRegion = toVec3i(Vec3(patResoLen) * sizeFactor);
	if (is2D) desRegion.z = 1;
	FluidSolver tmpSolver(desRegion, is2D ? 2 : 3);
	Grid<Vec3> regionG(&tmpSolver, false);

	// no need to parallel, due to I/O access
	FOR_PARTS(*this) {// IO function, parallel won't be good
		if (!isActive(idx)) continue;
		if (!doFadOut && isFadingOut(idx)) continue;

		saveLocalPatchMACGridTempl<S>(idx, dataG, regionG, *this, preIndexGrid, preTetGrid);
		knUndeformV<S>(regionG, *this, idx, preIndexGrid);// vector undeform!!
		if (rescale) {\
			Real maxV = regionG.getMax(); // normSquare
			regionG.multConst( Vec3( (1.0) / sqrt(maxV) ) );
		}

		if (step) {// quantize! 
			quantizeGridV(regionG, *step);
		}

		std::string unistr, patchname;
		getPath(mData[idx].uniqID, mData[idx].lifet, true, dirPath, unistr, patchname);

		if (ppmFlag) {
			if (is2D) {
				SimpleImage outputPPM;
				outputPPM.init(regionG.getSizeX(), regionG.getSizeY());
				FOR_IJK(regionG) {
					outputPPM.get(i, j) = regionG(i, j, k);
				}
				outputPPM.writePpm(unistr + patchname + std::string(".ppm"));
			} else { /* ???? 
				| XY	| XZ	|YZ		|
				|z=0	|y=0	|x=0	|
				|z=mid	|y=mid	|x=mid	|
				|z=end	|y=end	|x=end	|  */
				SimpleImage outputPPM;
				Vec3i s = regionG.getSize();
				int imagY = std::max(s[1], s[2]);
				outputPPM.init(s[0]*2+ s[2], 3 * imagY);
				FOR_IJK(regionG) {
					if( i == 0 )
						outputPPM.get(s[0] * 2 + k, j) = regionG(i, j, k);
					if (i == s[0] - 1)
						outputPPM.get(s[0] * 2 + k, imagY * 2 + j) = regionG(i, j, k);
					if (i == s[0]/2)
						outputPPM.get(s[0] * 2 + k, imagY + j) = regionG(i, j, k);

					if (j == 0)
						outputPPM.get(s[0] + i, k) = regionG(i, j, k);
					if (j == s[1] - 1)
						outputPPM.get(s[0] + i, imagY * 2 + k) = regionG(i, j, k);
					if (j == s[1] / 2)
						outputPPM.get(s[0] + i, imagY + k) = regionG(i, j, k);

					if (k == 0)
						outputPPM.get(i, j) = regionG(i, j, k);
					if (k == s[2] - 1)
						outputPPM.get(i, imagY * 2 + j) = regionG(i, j, k);
					if (k == s[2] / 2)
						outputPPM.get(i, imagY + j) = regionG(i, j, k);
				}
				outputPPM.writePpm(unistr + patchname + std::string(".ppm"));
			}
		}
		regionG.save(unistr + patchname + std::string(".uni"));
	}
}

template<class S>
void PatchSystem<S>::saveLocalPatchMACCurl(MACGrid & dataG, std::string dirPath, bool rescale, Real * step, bool ppmFlag, bool doFadOut)
{// acceleration is STRONGLY recommended!
	bool is2D = (getParent()->is2D());
	Vec3 sizeFactor = calcGridSizeFactor(dataG.getSize(), getParent()->getGridSize());
	Vec3i desRegion = toVec3i(Vec3(patResoLen) * sizeFactor);
	if (is2D) desRegion.z = 1;
	FluidSolver tmpSolver(desRegion, is2D ? 2 : 3);
	Grid<Vec3> regionG(&tmpSolver, false);
	Grid<Vec3> curlG(&tmpSolver, false);
	// no need to parallel, due to I/O access
	FOR_PARTS(*this) {// IO function, parallel won't be good
		if (!isActive(idx)) continue;
		if (!doFadOut && isFadingOut(idx)) continue;

		saveLocalPatchMACGridTempl<S>(idx, dataG, regionG, *this, preIndexGrid, preTetGrid);
		knUndeformV<S>(regionG, *this, idx, preIndexGrid);// vector undeform!!
		CurlOp(regionG, curlG);

		if (rescale) {
			Real maxV = curlG.getMax(); // normSquare
			curlG.multConst(Vec3((1.0) / sqrt(maxV)));
		}

		if (step) {// quantize!, work for both 2D and 3D
			quantizeGridV(curlG, *step);
		}

		std::string unistr, patchname;
		getPath(mData[idx].uniqID, mData[idx].lifet, true, dirPath, unistr, patchname);

		if (ppmFlag) {
			if (is2D) {
				SimpleImage outputPPM;
				outputPPM.init(curlG.getSizeX(), curlG.getSizeY());
				FOR_IJK(curlG) {
					Real curZ = curlG(i, j, k)[2];
					outputPPM.get(i, j) = Vec3(0.0);
					if (curZ < 0.0)
						outputPPM.get(i, j)[0] = -curZ;
					else
						outputPPM.get(i, j)[1] = curZ;
				}
				outputPPM.writePpm(unistr + patchname + std::string(".ppm"));				
			} else { /* ???? 
				| XY	| XZ	|YZ		|
				|z=0	|y=0	|x=0	|
				|z=mid	|y=mid	|x=mid	|
				|z=end	|y=end	|x=end	|  */

				SimpleImage outputPPM;
				Vec3i s = curlG.getSize();
				int imagY = std::max(s[1], s[2]);
				outputPPM.init(s[0] * 2 + s[2], 3 * imagY);
				FOR_IJK(curlG) {
					if (i == 1)
						outputPPM.get(s[0] * 2 + k, j) = curlG(i, j, k);
					if (i == s[0] - 2)
						outputPPM.get(s[0] * 2 + k, imagY * 2 + j) = curlG(i, j, k);
					if (i == s[0] / 2)
						outputPPM.get(s[0] * 2 + k, imagY + j) = curlG(i, j, k);

					if (j == 1)
						outputPPM.get(s[0] + i, k) = curlG(i, j, k);
					if (j == s[1] - 2)
						outputPPM.get(s[0] + i, imagY * 2 + k) = curlG(i, j, k);
					if (j == s[1] / 2)
						outputPPM.get(s[0] + i, imagY + k) = curlG(i, j, k);

					if (k == 1)
						outputPPM.get(i, j) = curlG(i, j, k);
					if (k == s[2] - 2)
						outputPPM.get(i, imagY * 2 + j) = curlG(i, j, k);
					if (k == s[2] / 2)
						outputPPM.get(i, imagY + j) = curlG(i, j, k);
				}
				outputPPM.writePpm(unistr + patchname + std::string(".ppm"));
			}
		}

		if (is2D) {
			Grid<Real> zregionG(&tmpSolver, false);
			GetComponent(curlG, zregionG, 2);
			zregionG.save(unistr + patchname + std::string(".uni"));
		}else
			curlG.save(unistr + patchname + std::string(".uni"));
	}
}

template<class S>
int PatchSystem<S>::saveLocalPatchNumpyReal(Grid<Real>& dataG, PyArrayContainer patchGrids, PyArrayContainer patchDict, Vec3i desRegion, bool doFadOut){ // acceleration is recommended!
	// assert patchGrids.size > bgid * desRegion.x * regionGdesRegiony * desRegion.z
	// assert patchDict.size >= mData.size()
	bool is2D = (getParent()->is2D());
	if(desRegion.x <= 0)
		desRegion = toVec3i(Vec3(patResoLen) * calcGridSizeFactor(dataG.getSize(), getParent()->getGridSize()));
	if (is2D) desRegion.z = 1;
	FluidSolver tmpSolver(desRegion, is2D ? 2 : 3);
	Grid<Real> regionG(&tmpSolver, false); // reuse one grid for efficiency

	int bgid = 0;
	FOR_PARTS(*this) {// list generation
		if (!isActive(idx)) {
			(reinterpret_cast<int*>(patchDict.pData))[idx] = -1;
			continue;
		}

		if (!doFadOut && isFadingOut(idx)) {
			(reinterpret_cast<int*>(patchDict.pData))[idx] = -1;
			continue;
		}

		(reinterpret_cast<int*>(patchDict.pData))[idx] = bgid++;
	}

	#pragma omp for // use another KERNEL(pts) leads to endless compiling
	for (IndexInt idx = 0; idx< size(); idx++) {
		if (!isActive(idx)) continue;
		if (!doFadOut && isFadingOut(idx)) continue;
		knSaveLocalPatchGridNumpy<S, Real>(regionG, dataG, desRegion, *this, idx,
			patchGrids, patchDict, preIndexGrid, preTetGrid);
	}
	
	//do rescaling with numpy array directly in python, faster
	return bgid;
}

template<class S>
int PatchSystem<S>::saveLocalPatchNumpyMAC(MACGrid& dataG, PyArrayContainer patchGrids, PyArrayContainer patchDict, Vec3i desRegion, bool doFadOut) { // acceleration is recommended!
	// assert patchGrids.size > bgid * regionG.x * regionG.y * regionG.z * 3
	// assert patchDict.size >= mData.size()
	bool is2D = (getParent()->is2D());
	if(desRegion.x <= 0)
		desRegion = toVec3i(Vec3(patResoLen) * calcGridSizeFactor(dataG.getSize(), getParent()->getGridSize()));
	if (is2D) desRegion.z = 1;
	FluidSolver tmpSolver(desRegion, is2D ? 2 : 3);
	MACGrid regionG(&tmpSolver, false); // reuse one grid for efficiency

	int bgid = 0;
	FOR_PARTS(*this) {// list generation
		if (!isActive(idx)) {
			(reinterpret_cast<int*>(patchDict.pData))[idx] = -1;
			continue;
		}
		if (!doFadOut && isFadingOut(idx)) {
			(reinterpret_cast<int*>(patchDict.pData))[idx] = -1;
			continue;
		}
		(reinterpret_cast<int*>(patchDict.pData))[idx] = bgid++;
	}

	#pragma omp for // use another KERNEL(pts) leads to endless compiling
	for (IndexInt idx = 0; idx< size(); idx++) {
		if (!isActive(idx)) continue;
		if (!doFadOut && isFadingOut(idx)) continue;
		knSaveLocalPatchMACGridNumpy<S>(regionG, dataG, desRegion, *this, idx,
			patchGrids, patchDict, preIndexGrid, preTetGrid);
		knUndeformVNumpy<S>(regionG, patchGrids, patchDict, *this, idx, preIndexGrid);// vector undeform!!
	}

	//do rescaling with numpy array directly in python, faster
	return bgid;
}


KERNEL(bnd = 1)
void CurlOpNumpy(const Grid<Vec3>& grid, PyArrayContainer dst, int gid) {

	int idx2 = grid.index(i, j, k) + gid * (grid.getSizeX() * grid.getSizeY() * grid.getSizeZ());

	Vec3 v = Vec3(0., 0.,
		0.5*((grid(i + 1, j, k).y - grid(i - 1, j, k).y) - (grid(i, j + 1, k).x - grid(i, j - 1, k).x)));
	if (grid.is3D()) {
		v[0] = 0.5*((grid(i, j + 1, k).z - grid(i, j - 1, k).z) - (grid(i, j, k + 1).y - grid(i, j, k - 1).y));
		v[1] = 0.5*((grid(i, j, k + 1).x - grid(i, j, k - 1).x) - (grid(i + 1, j, k).z - grid(i - 1, j, k).z));

		Vec3* dataidx2 = NULL;
		dst.get(idx2, dataidx2);
		if (dataidx2) (*dataidx2) = v;
	} else {
		Real* dataidx2 = NULL;
		dst.get(idx2, dataidx2);
		if (dataidx2) (*dataidx2) = v.z;
	}
};

template<class S>
int PatchSystem<S>::saveLocalPatchNumpyCurl(MACGrid& dataG, PyArrayContainer patchGrids, PyArrayContainer patchDict, Vec3i desRegion, bool doFadOut) { // acceleration is recommended!
	// assert patchGrids.size > bgid * desRegion.x * desRegion.y * desRegion.z * (is2D ? 1:3)
	// assert patchDict.size >= mData.size()
	bool is2D = (getParent()->is2D());
	if(desRegion.x <=0) desRegion = toVec3i(Vec3(patResoLen) * calcGridSizeFactor(dataG.getSize(), getParent()->getGridSize()));
	if (is2D) desRegion.z = 1;
	FluidSolver tmpSolver(desRegion, is2D ? 2 : 3);
	
	int bgid = 0;
	std::vector<Grid<Vec3>*> localGs;
	FOR_PARTS(*this) {// list generation
		if (!isActive(idx)) {
			(reinterpret_cast<int*>(patchDict.pData))[idx] = -1;
			continue;
		}
		if (!doFadOut && isFadingOut(idx)) {
			(reinterpret_cast<int*>(patchDict.pData))[idx] = -1;
			continue;
		}
		(reinterpret_cast<int*>(patchDict.pData))[idx] = bgid++;
		localGs.push_back(new MACGrid(&tmpSolver, false));
	}

	#pragma omp for // use another KERNEL(pts) leads to endless compiling
	for (IndexInt idx = 0; idx< size(); idx++) {
		if (!isActive(idx)) continue;
		if (!doFadOut && isFadingOut(idx)) continue;
		int gid = (reinterpret_cast<int*>(patchDict.pData))[idx];

		knSaveLocalPatchMACGrid<S>((*localGs[gid]), dataG, desRegion, *this, idx, preIndexGrid, preTetGrid);
		knUndeformV<S>((*localGs[gid]), *this, idx, preIndexGrid);// vector undeform!!
		CurlOpNumpy((*localGs[gid]), patchGrids, gid);
	}
	for (int gid = 0; gid < bgid; gid++) {
		delete localGs[gid];
	}
	//do rescaling with numpy array directly in python, faster
	return bgid;
}


KERNEL(pts) template<class S>
void knParUpdate(PatchSystem<S>& pp) {
	if (!pp.isActive(idx)) return;
	pp.addParLifeT(idx);
	pp.removeNewFlag(idx);
}
template<class S>
void PatchSystem<S>::updateParts(bool compress)
{
	knParUpdate<S>(*this);
	if (compress) {
		doCompress();
	}
}

template<class S>
void PatchSystem<S>::killBad(Real maxDefE, PyArrayContainer* BadList, Grid<Real>* den) {
	if (BadList) {
		int badSZ = BadList->TotalSize;
		for (int bi = 0; bi < badSZ; ++bi) {
			int idx = (reinterpret_cast<int*>(BadList->pData))[bi];
			if (!isActive(idx)) continue;
			kill(idx);
		}
	}

	int canID = -1, minID = -1;
	Real canMinError = 4.0;
	Vec3 factor(0.0);
	Grid<Real>* denMask = NULL;
	if (den) {
		denMask = new Grid<Real>(den->getParent(), false);
		blurGrid<Real>(*den, *denMask, patResoLen);
		factor = calcGridSizeFactor(denMask->getSize(), this->getParent()->getGridSize());
	}

	FOR_PARTS(*this) {
		if (!isActive(idx)) continue;
		// kill outsiders
		if (denMask) {
			if (denMask->getInterpolated(mData[idx].pos * factor) < 1e-6f)
				kill(idx); 
		}

		// fading out too deformed ones
		if (mData[idx].defoError > maxDefE) {
			if ((mData[idx].flag & PFADINGIN) != 0) mData[idx].flag -= PFADINGIN;
			mData[idx].flag |= PFADINGOUT;
		}
	}
	
	if (denMask) {
		delete denMask;
	}
}


template<class S>
void PatchSystem<S>::meshView(Mesh& mesh, Vec3 factor) {
	mesh.clear();
	FOR_PARTS(*this) {
		if (!isActive(idx)) continue;
		//if (idx != 5) continue; // debug
		mData[idx].defoData.meshView(mesh, factor);
	}
}

// kernelW stands for the blur diameter. sigma stands for the standard deviation "removed" by blur
// With no sigm given, a proper one is calculated based on the kernelW
PYTHON() void blurRealGrid(Grid<Real>& source, Grid<Real>& target, Real kernelW = -0.1, Real sigm = -0.1)
{
	if (sigm > VECTOR_EPSILON && kernelW < VECTOR_EPSILON) kernelW = sigm * 6.0;
	blurGrid<Real>(source, target, kernelW, sigm);
}

PYTHON() void blurMacGrid(MACGrid& source, MACGrid& target, Real kernelW = -0.1, Real sigm = -0.1)
{
	if (sigm > VECTOR_EPSILON && kernelW < VECTOR_EPSILON) kernelW = sigm * 6.0;
	blurGrid<Vec3>(source, target, kernelW, sigm);
}


// explicit instantiation
template class PatchSystem<PatchData>;
}