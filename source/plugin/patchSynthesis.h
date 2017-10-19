/******************************************************************************
*
* MantaFlow fluid solver framework
* Copyright 2011 Tobias Pfaff, Nils Thuerey
*
* This program is free software, distributed under the terms of the
* GNU General Public License (GPL)
* http://www.gnu.org/licenses
*
* Deformation-limiting Patch System by Mengyu Chu (mengyu.chu@tum.de)
*
* This is an implementation of deformation-limiting patch motion [1].
*
* [1] Data-Driven Synthesis of Smoke Flows with CNN-based Feature Descriptors,
*     Mengyu Chu and Nils Thuerey. 2017. ACM Trans. Graph. 36, 4 (2017).
*
******************************************************************************/

#ifndef PATCHSYN_H
#define PATCHSYN_H

#include "patchAdv.h"
#include <fstream> 

namespace Manta {

struct parInfo {// info for anticipation in backward
	Real  parShowTime;
	PatchData parData;
	int parMatchPat; // matched ID init
	Vec3  parMatchErrorMinMax;

	parInfo(Real t, PatchData copyData, int matchID, Real matchError, Real matchMin, Real matchMax):
		parShowTime(t), parData(copyData), parMatchPat(matchID), 
		parMatchErrorMinMax(matchError, matchMin, matchMax){}
	parInfo() :
		parShowTime(0.0), parData(), parMatchPat(-1), parMatchErrorMinMax(0.0) {
		parData.flag = -1;
	}
};

PYTHON() class PatchSynSystem : public PatchSystem<PatchData> {
// class PatchSystem focuses on patch advection, coordinates transfer between patch and world 
// class PatchSynSystem
// has more fading control, including patch anticipation, fading out bad patches ,...
// support repository with matching ID and matching quality
public:
	PYTHON() PatchSynSystem(FluidSolver* parent, int subdiv = 2, int baseRes = 12, 
		int jN = 1, bool anticipate = true)
		: PatchSystem<PatchData>(parent, subdiv, baseRes, jN), matchID(parent), matchError(parent), 
		matchMin(parent), matchMax(parent),
		mAnticipate(anticipate){
		/* // debug!!
		Patch_DefoData testData;
		testData.init(patRest, &(patRest.restPoints));
		std::vector<Vec3> NewP(patRest.restPoints);
		for (int i = 0; i < Newp.size(); ++i) {
		NewP[i] = patRest.restPoints[i] * 2.0 + Vec3(0.5, 0.5, 0.5);
		}
		testData.adjustDefoInPlace(NewP,0.002);
		*/
		this->mAllowCompress = false;
		this->registerPdataReal(&matchError);
		this->registerPdataReal(&matchMin);
		this->registerPdataReal(&matchMax);
		this->registerPdataInt(&matchID);
	}
		
	PYTHON() void initNewPatchInfo(Real fadWei = 1.0);
	PYTHON() void getMatchList(bool getFadOut = false, PyArrayContainer* matchList = NULL,
		PyArrayContainer* tarMin = NULL, PyArrayContainer* tarMax = NULL);
	PYTHON() void getFadingWeiList(PyArrayContainer fadList);
	PYTHON() void setMatchList(PyArrayContainer matchList, PyArrayContainer* NmatchError = NULL,
		PyArrayContainer* tarMin = NULL, PyArrayContainer* tarMax = NULL);
	// similar to killBad, fade out instead of kill
	PYTHON() void removeBad(Real maxDefE, PyArrayContainer* BadList = NULL, Grid<Real>* den = NULL);
	PYTHON() void updateFading(Real fadT);// update forward (fadT>0) or backward (fadT<0)

	PYTHON() void anticipateAdd(Real framet);
	PYTHON() void anticipateStore(Real framet);
	// just for debug info
	PYTHON() void printParts(int start = -1, int stop = -1, bool printIndex = false);

	PYTHON() void clearParts() {
		FOR_PARTS(*this) {
			kill(idx);
		}
		doCompress();
		matchID.resize(0);
		matchError.resize(0);
		matchMin.resize(0);
		matchMax.resize(0);
	}

	PYTHON() void printLog(std::string filepath);

	ParticleDataImpl<int> matchID;
	ParticleDataImpl<Real> matchError;
	ParticleDataImpl<Real> matchMin;
	ParticleDataImpl<Real> matchMax;

protected:
	bool mAnticipate;
	

	std::vector<parInfo> patchStore;// for anticipation

};

}
#endif