#include "patchSynthesis.h"

namespace Manta{

void PatchSynSystem::printParts(int start, int stop, bool printIndex)
{
	std::ostringstream sstr;
	IndexInt s = (start>0 ? start : 0                      );
	IndexInt e = (stop>0  ? stop  : (IndexInt)mData.size() );
	s = Manta::clamp(s, (IndexInt)0, (IndexInt)mData.size());
	e = Manta::clamp(e, (IndexInt)0, (IndexInt)mData.size());

	for(IndexInt i=s; i<e; ++i) {
		if(printIndex) sstr << i<<": ";
		sstr<<mData[i].pos<<" "<<mData[i].flag << " " << mData[i].fadeWei << " " << matchID[i] << " " << matchError[i] <<"\n";
	} 
	debMsg( sstr.str() , 1 );
}

KERNEL(pts) void knGetMatchList(PatchSynSystem& pp, bool getFadOut,
	PyArrayContainer* matchList, PyArrayContainer* tarMin, PyArrayContainer* tarMax) {
	if (!pp.isActive(idx)) return;
	if (getFadOut == false && pp.isFadingOut(idx)) return;

	if (matchList)(reinterpret_cast<int*>(matchList->pData))[idx] = pp.matchID[idx];
	if (tarMin) (reinterpret_cast<Real*>(tarMin->pData))[idx] = pp.matchMin[idx];
	if (tarMax) (reinterpret_cast<Real*>(tarMax->pData))[idx] = pp.matchMax[idx];
}

void PatchSynSystem::getMatchList(bool getFadOut, PyArrayContainer* matchList, PyArrayContainer* tarMin, PyArrayContainer* tarMax)
{
	matchID.resize(size());
	matchMin.resize(size());
	matchMax.resize(size());

	knGetMatchList(*this, getFadOut, matchList, tarMin, tarMax);
}

KERNEL(pts) void knGetFadingWeiList(PatchSynSystem& pp, PyArrayContainer& fadList) {
	if (!pp.isActive(idx)) return;
	(reinterpret_cast<Real*>(fadList.pData))[idx] = pp.getFading(idx);
}

void PatchSynSystem::getFadingWeiList(PyArrayContainer fadList) {
	knGetFadingWeiList(*this, fadList);
}

KERNEL(pts) void knSetMatchList(PatchSynSystem& pp, PyArrayContainer& matchList, PyArrayContainer* NmatchError,
	PyArrayContainer* tarMin, PyArrayContainer* tarMax) {
	if (!pp.isActive(idx)) return;

	pp.matchID[idx] = (reinterpret_cast<int*>(matchList.pData))[idx];
	if (pp.isNewPatch(idx) && (pp.matchID[idx] < 0)) {
		pp.kill(idx);
		return;
	}
	if (NmatchError) pp.matchError[idx] = (reinterpret_cast<Real*>(NmatchError->pData))[idx];

	if (!pp.isNewPatch(idx)) return;
	if (tarMin) pp.matchMin[idx] = (reinterpret_cast<Real*>(tarMin->pData))[idx];
	if (tarMax) pp.matchMax[idx] = (reinterpret_cast<Real*>(tarMax->pData))[idx];
}

void PatchSynSystem::setMatchList(PyArrayContainer matchList, PyArrayContainer* NmatchError,
	PyArrayContainer* tarMin, PyArrayContainer* tarMax) {
	matchID.resize(size());
	matchError.resize(size());
	matchMin.resize(size());
	matchMax.resize(size());
	knSetMatchList(*this, matchList, NmatchError, tarMin, tarMax);
}

KERNEL(pts) void knInitNewMatchInfo(PatchSynSystem& pp, Real fadW) {
	if (!pp.isActive(idx)) return;
	if (!pp.isNewPatch(idx)) return;
	pp.matchID[idx] = -1;
	pp.matchError[idx] = 4.0;
	pp.matchMin[idx] = 0.0;
	pp.matchMax[idx] = 1.0;

	pp.setFading(idx, fadW);

	if (fadW < 1.0 - VECTOR_EPSILON)
		pp.setFadingIn(idx);
}

void PatchSynSystem::initNewPatchInfo(Real fadWei) {
	matchID.resize(size());
	matchError.resize(size());
	matchMin.resize(size());
	matchMax.resize(size());
	knInitNewMatchInfo(*this, fadWei);
}

void PatchSynSystem::removeBad(Real maxDefE, PyArrayContainer* BadList, Grid<Real>* den) {
	if (BadList) {
		int badSZ = BadList->TotalSize;
		for (int bi = 0; bi < badSZ; ++bi) {
			int idx = (reinterpret_cast<int*>(BadList->pData))[bi];
			if (!isActive(idx)) continue;
			if (isNewPatch(idx)) {
				kill(idx);
				continue;
			}
			if ((mData[idx].flag & PFADINGIN) != 0) mData[idx].flag -= PFADINGIN;
			mData[idx].flag |= PFADINGOUT;
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
		// kill time out
		if (isFadingOut(idx)) {
			//if (mData[idx].fadeWei == 0.0)
			//	kill(idx);
			continue;
		}
		// kill bad candidates
		if (isNewPatch(idx)) {
			if (canID == mData[idx].uniqID) {
				if (matchError[idx] < canMinError) {//right one!
					if (minID >= 0) kill(minID);
					canMinError = matchError[idx];
					minID = idx;
				} else {
					kill(idx);
					continue;
				}
			} else {
				canID = mData[idx].uniqID;
				canMinError = matchError[idx];
				minID = idx;
			}
		} else {
			// kill outsiders
			if (denMask) {
				if((!denMask->isInBounds(toVec3i(mData[idx].pos * factor))) ||
					(denMask->getInterpolated(mData[idx].pos * factor) < 1e-6f) )
					kill(idx); // not fading out, but to be killed directly
			}

			// fading out too deformed ones
			if (mData[idx].defoError > maxDefE) {
				if ((mData[idx].flag & PFADINGIN) != 0) mData[idx].flag -= PFADINGIN;
				mData[idx].flag |= PFADINGOUT;
			}
		}
	}

	if (denMask) {
		delete denMask;
	}
}

void PatchSynSystem::updateFading(Real fadT) {
	FOR_PARTS(*this) {
		if (!isActive(idx)) continue;
		if (isFadingIn(idx)) {
			this->mData[idx].fadeWei += fadT;
		}
		else if (isFadingOut(idx)) {
			this->mData[idx].fadeWei -= fadT;
		}
		else
			continue;

		if (this->mData[idx].fadeWei <= VECTOR_EPSILON) {
			this->mData[idx].fadeWei = 0.0;
			kill(idx);
		} else if (this->mData[idx].fadeWei > 1.0 - VECTOR_EPSILON) {
			this->mData[idx].fadeWei = 1.0;
			if ((mData[idx].flag & PFADINGIN) != 0) mData[idx].flag -= PFADINGIN;
			if ((mData[idx].flag & PFADINGOUT) != 0) mData[idx].flag -= PFADINGOUT;
		}
	}
}

void PatchSynSystem::printLog(std::string filepath) {

	std::ostringstream sstr;
	sstr << "ID\tFadingW\tFadingS\tMatchID\tMatchEr\tPos\n";
	FOR_PARTS(*this) {
		if (!isActive(idx)) continue;
		sstr << idx << "\t" << mData[idx].fadeWei << "\t";
		if ((mData[idx].flag & PFADINGIN) != 0) sstr << "In\t";
		else if ((mData[idx].flag & PFADINGOUT) != 0) sstr << "Out\t";
		else sstr << "OK\t";
		
		sstr << matchID[idx] << "\t" << matchError[idx] << "\t";

		sstr << this->getPos(idx) <<"\n"; 
	}

	std::ofstream fout(filepath.c_str(), std::ofstream::out | std::ofstream::app);
	if (fout.good()) {
		fout << sstr.str();
		fout.close();
	}
	else
		debMsg(sstr.str(), 1);
}

void PatchSynSystem::anticipateStore(Real framet) {// can not parallel
	FOR_PARTS(*this) {
		if (!isActive(idx)) continue;
		if (!isNewPatch(idx)) continue;
		patchStore.push_back(parInfo(framet, mData[idx], matchID[idx], matchError[idx], matchMin[idx], matchMax[idx]));
	}
}

void PatchSynSystem::anticipateAdd(Real framet) {// can not parallel
	int lastOne;
	for (lastOne = patchStore.size() - 1; 
		lastOne >= 0 && patchStore[lastOne].parShowTime > (framet - 0.5);
		lastOne--) {

		if (patchStore[lastOne].parShowTime > (framet + 0.5)) continue;
		IndexInt resultID = ParticleSystem<PatchData>::add(patchStore[lastOne].parData);
		mData[resultID].flag = 0|PNEW;
		mData[resultID].flag |= PFADINGIN;
		mData[resultID].fadeWei = 1.0f;

		matchID.resize(resultID+1);
		matchError.resize(resultID + 1);
		matchMin.resize(resultID + 1);
		matchMax.resize(resultID + 1);

		matchID[resultID] = patchStore[lastOne].parMatchPat;
		matchError[resultID] = patchStore[lastOne].parMatchErrorMinMax.x;
		matchMin[resultID] = patchStore[lastOne].parMatchErrorMinMax.y;
		matchMax[resultID] = patchStore[lastOne].parMatchErrorMinMax.z;
	}
	patchStore.resize(lastOne + 1);
}


KERNEL() void knSynthesisScale(Grid<Real>& synResult, Grid<Real>& weiG, Grid<Real>& base, Grid<Real>& baseBlur, Real cutoff = 0.01) {
	Vec3 factorW = calcGridSizeFactor(weiG.getSize(), synResult.getSize());
	Vec3 factorB = calcGridSizeFactor(base.getSize(), synResult.getSize());

	Vec3 pos(i + 0.5, j + 0.5, k + 0.5);
	Real posM = baseBlur.getInterpolatedHi(pos*factorB, 1);
	if (posM < VECTOR_EPSILON) {
		synResult(i, j, k) = 0.0;
		return;
	}

	Real posW = weiG.getInterpolatedHi(pos*factorW, 1);
	if (posW > 1.0)
		synResult(i, j, k) = synResult(i, j, k) / posW;
	else if (posW > 0.0)
		synResult(i, j, k) = synResult(i, j, k) + base.getInterpolatedHi(pos*factorB, 1) * (1.0 - posW);
	else
		synResult(i, j, k) = base.getInterpolatedHi(pos*factorB, 1);

	if (posM < cutoff) {
		synResult(i, j, k) *= (posM / cutoff);
	}
}

PYTHON() void synthesisScale(Grid<Real>& synResult, Grid<Real>& weiG, Grid<Real>& base, Real cutoff = 0.01) {
	Grid<Real> baseMask(base);
	blurGrid(base, baseMask, 9.0);
	knSynthesisScale(synResult, weiG, base, baseMask, cutoff);
}

extern void interpolateGrid(Grid<Real>& target, Grid<Real>& source, Vec3 scale = Vec3(1.), Vec3 offset = Vec3(0.), Vec3i size = Vec3i(-1, -1, -1), int orderSpace = 1);
extern void copyGridToArrayReal(const Grid<Real>& source, PyArrayContainer target);
extern void interpolateGridVec3(Grid<Vec3>& target, Grid<Vec3>& source, Vec3 scale = Vec3(1.), Vec3 offset = Vec3(0.), Vec3i size = Vec3i(-1, -1, -1), int orderSpace = 1);
extern void copyGridToArrayVec3(const Grid<Vec3>& source, PyArrayContainer target);

PYTHON() void numpyGridResize(std::string path, PyArrayContainer targetArray, Vec3i sz_in, Vec3i sz_out, int data_type) {
	int dim = 3;
	if (sz_in.z == 1 && sz_out.z == 1)
		dim = 2;

	FluidSolver tmpInput(sz_in, dim, -1);
	FluidSolver tmpOutput(sz_out, dim, -1);

	if (data_type == 1) { //Real
		Grid<Real> denInput(&tmpInput, false);
		Grid<Real> denOutput(&tmpOutput, false);
		denInput.load(path);
		interpolateGrid(denOutput, denInput);
		copyGridToArrayReal(denOutput, targetArray);
	}
	else { //Vec3
		Grid<Vec3> velInput(&tmpInput, false);
		Grid<Vec3> velOutput(&tmpOutput, false);
		velInput.load(path);
		interpolateGridVec3(velOutput, velInput);
		copyGridToArrayVec3(velOutput, targetArray);
	}
}


}