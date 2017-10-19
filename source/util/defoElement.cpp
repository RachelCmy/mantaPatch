/******************************************************************************
*
* MantaFlow fluid solver framework
*
* Patch deformation
*
******************************************************************************/
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "defoElement.h"

using namespace Eigen;

namespace Manta {

//! discretize patch (ie "quad") with 5 tets
const int tetIndices[5 * 4] = {
	0, 1, 2, 4,
	1, 2, 4, 7,
	1, 2, 7, 3,
	4, 7, 2, 6,
	1, 4, 5, 7 };

//! undeformed coords
const Vec3 stateRestUnit[8] = {
	Vec3(0.0, 0.0, 0.0),
	Vec3(1.0, 0.0, 0.0),
	Vec3(0.0, 1.0, 0.0),
	Vec3(1.0, 1.0, 0.0),
	Vec3(0.0, 0.0, 1.0),
	Vec3(1.0, 0.0, 1.0),
	Vec3(0.0, 1.0, 1.0),
	Vec3(1.0, 1.0, 1.0) };

// imaginary tetrahedron neighbours of current vertex
const int neibIdx[8 * 3] = {
	1, 4, 2, // tetrahedron 0, 0,1,4,2
	5, 0, 3, // tetrahedron 1, 1,5,0,3
	6, 3, 0, // tetrahedron 2, 2,...
	2, 7, 1, // tetrahedron 3, 3,...
	0, 5, 6, // tetrahedron 4, 4,...
	4, 1, 7, // tetrahedron 5, 5,...
	7, 2, 4, // tetrahedron 6, 6,...
	3, 6, 5	 // tetrahedron 7, 7,...
};

// 12 edges in a cube, e01, e02,...
const int edgeIdx[12 * 2] = {
	0,1,	0,2,	0,4,	1,3,
	1,5,	2,3,	2,6,	3,7,
	4,5,	4,6,	5,7,	6,7
};

//! compute inverse transformation for a list of tets, state.size() = list.size()
void buildTrafoList(const std::vector<Vec3>& state, std::vector<Mat4>& list)
{
	Mat4 P;
	Mat4 Pinv;
	list.clear();
	for (int i = 0; i<state.size(); i += 4) {
		Vec3 p[4] = { state[i + 0], state[i + 1], state[i + 2], state[i + 3] };

		// compute tet deformation inverse
		P = Mat4(0.);
		P(0, 0) = p[1].x - p[0].x;
		P(0, 1) = p[2].x - p[0].x;
		P(0, 2) = p[3].x - p[0].x;
		P(1, 0) = p[1].y - p[0].y;
		P(1, 1) = p[2].y - p[0].y;
		P(1, 2) = p[3].y - p[0].y;
		P(2, 0) = p[1].z - p[0].z;
		P(2, 1) = p[2].z - p[0].z;
		P(2, 2) = p[3].z - p[0].z;
		P(3, 3) = 1.;

		Pinv = P;
		if (!Pinv.invert()) {
			// fallback if invert fails!
			Pinv = Mat4(0.);
			Pinv(0, 0) = -1000.; // "marker"
			debMsg("Warning - invalid defo " << (i / 4), 1);
		}
		list.push_back(Pinv);
	}
}

// Matrix for scaling control, depending on the subd. Just build once, no need to rebuild
void HMatrixGen(int subd, SparseMatrix<Real>& HS) {
	// assert size matches, HS.rows() = m;
	const int n = 2 + subd;
	int m = (n)*(n)*(n);

	HS.setZero();
	HS.reserve(VectorXi::Constant(m, 7)); // each row has at most 7 entry, at least 4 entry

	int p[8] = { 0 };
	//VectorXf curVn(m);//debug
	for (int cubk = 0; cubk < subd + 1; ++cubk)
	for (int cubj = 0; cubj < subd + 1; ++cubj)
	for (int cubi = 0; cubi < subd + 1; ++cubi) {// go through all cube
		// current cube vertex, 0 - (pSize - 1)	
		p[0] = cubk*n*n + cubj*n + cubi;
		p[1] = p[0] + 1; p[2] = p[0] + n;
		p[3] = p[2] + 1;
		p[4] = p[0] + n*n;
		p[5] = p[4] + 1; p[6] = p[4] + n;
		p[7] = p[6] + 1;

		for (int eij = 0; eij < 12; ++eij) { // for every edge in the cube
			int v[2] = { p[edgeIdx[eij * 2 + 0]] , p[edgeIdx[eij * 2 + 1]] };
			HS.coeffRef(v[0], v[0]) += 1.0;
			HS.coeffRef(v[1], v[1]) += 1.0;

			HS.coeffRef(v[0], v[1]) -= 1.0;
			HS.coeffRef(v[1], v[0]) -= 1.0;
		}
	}
	HS.makeCompressed();// optional
}

void Patch_RestInfo::init(int subdiv) {
	subdN = subdiv;
	const int numTetsCube = 5;
	if (subdiv == 0) { // special case
		for (int c = 0; c<8; ++c) {
			restPoints.push_back(stateRestUnit[c]);
		}
		for (int c = 0; c<numTetsCube * 4; ++c) {
			currIndices.push_back(tetIndices[c]);
		}
	} else {
		Real dinv = 1. / Real(1 + subdiv);

		// create rest state points
		for (int k = 0; k<2 + subdiv; ++k)
		for (int j = 0; j<2 + subdiv; ++j)
		for (int i = 0; i<2 + subdiv; ++i) {
			Vec3 pn = Vec3(i, j, k) * dinv;
			restPoints.push_back(pn);
		}

		// add indices
		for (int k = 0; k<1 + subdiv; ++k)
		for (int j = 0; j<1 + subdiv; ++j)
		for (int i = 0; i<1 + subdiv; ++i) {
			// tets that share face edges
			const int n = 2 + subdiv;
			int p0 = k*n*n + j*n + i;
			int p1 = p0 + 1;
			int p3 = (k + 1)*n*n + j*n + i;
			int p2 = p3 + 1;
			int p7 = (k + 1)*n*n + (j + 1)*n + i;
			int p6 = p7 + 1;
			int p4 = k*n*n + (j + 1)*n + i;
			int p5 = p4 + 1;
			if ((k + j + i) % 2 == 1) {
				currIndices.push_back(p2); currIndices.push_back(p1); currIndices.push_back(p6); currIndices.push_back(p3);
				currIndices.push_back(p6); currIndices.push_back(p3); currIndices.push_back(p4); currIndices.push_back(p7);
				currIndices.push_back(p4); currIndices.push_back(p1); currIndices.push_back(p6); currIndices.push_back(p5);
				currIndices.push_back(p3); currIndices.push_back(p1); currIndices.push_back(p4); currIndices.push_back(p0);
				currIndices.push_back(p6); currIndices.push_back(p1); currIndices.push_back(p4); currIndices.push_back(p3);
			} else {
				currIndices.push_back(p0); currIndices.push_back(p2); currIndices.push_back(p5); currIndices.push_back(p1);
				currIndices.push_back(p7); currIndices.push_back(p2); currIndices.push_back(p0); currIndices.push_back(p3);
				currIndices.push_back(p5); currIndices.push_back(p2); currIndices.push_back(p7); currIndices.push_back(p6);
				currIndices.push_back(p7); currIndices.push_back(p0); currIndices.push_back(p5); currIndices.push_back(p4);
				currIndices.push_back(p0); currIndices.push_back(p2); currIndices.push_back(p7); currIndices.push_back(p5);
			}
		}
	}
	currPoints = restPoints.size();
	currSize = currIndices.size();

	// init rest tets and volumes
	restTets.clear();
	defoInvRest.clear();

	for (int c = 0; c < currSize; ++c) {
		restTets.push_back(restPoints[currIndices[c]]);
	}

	buildTrafoList(restTets, defoInvRest);

	// init Matrix & solver for scaling control
	int m = (subdN + 2)*(subdN + 2)*(subdN + 2);
	SparseMatrix<Real> pH(m,m);
	HMatrixGen(subdN, pH);
	SimplicialCholesky< SparseMatrix<Real> >* solver = new SimplicialCholesky< SparseMatrix<Real> >();
	solver->compute(pH);
	if (solver->info() != Eigen::Success) {
		debMsg("not invertable!!", 0);
	}
	HS = solver;
}

Patch_RestInfo::~Patch_RestInfo() {
	if (HS) {
		SimplicialCholesky< SparseMatrix<Real> >* solver = (SimplicialCholesky< SparseMatrix<Real> >*) HS;
		delete solver;
		HS = NULL;
	}
}

// changed - init takes point list, but it's optional; if NULL use existing (possibly changed) points
// so with NULL, it's a "re-init"
void Patch_DefoData::init(Patch_RestInfo& rest, std::vector<Vec3> *_points /*= NULL*/) {
	// copy 8 corner points
	if (_points)
		points = *_points;

	tets.clear();
	// build tet lists
	for (int c = 0; c<rest.currSize; ++c) {
		const Vec3 p = points[rest.currIndices[c]];
		tets.push_back(p);

		// compute bbox
		if (c == 0) {
			bbmin = bbmax = p;
		}
		else {
			for (int d = 0; d<3; ++d) {
				if (p[d]<bbmin[d]) bbmin[d] = p[d];
				if (p[d]>bbmax[d]) bbmax[d] = p[d];
			}
		}
	}
	buildTrafoList(tets, defoInv);
	//debMsg("PDD init: "<<defoInv.size()<<" tets, bbox: "<<bbmin<<" to "<<bbmax <<"   "  ,1); 
	restInfo = &rest;
}

Vec3 Patch_DefoData::local2World(const Vec3& pPatch, int * hintTetID /*= NULL*/) const
{
	int idx = -1;
	Vec3 uvw;
	Real uv4;
	bool isInside;

	if (hintTetID != NULL && (*hintTetID) >= 0) { // validate hint
		idx = (*hintTetID);
		Vec3 pTr = pPatch -(*restInfo).restTets[4 * idx];
		uvw = (*restInfo).defoInvRest[idx] * pTr;
		uv4 = (1. - uvw[0] - uvw[1] - uvw[2]);

		if ((uvw[0]     < (0.f - VECTOR_EPSILON)) || (uvw[0]     > (1.f + VECTOR_EPSILON)) || (uvw[1] < (0.f - VECTOR_EPSILON)) ||
			(uvw[1]     > (1.f + VECTOR_EPSILON)) || (uvw[2]     < (0.f - VECTOR_EPSILON)) || (uvw[2] > (1.f + VECTOR_EPSILON)) ||
			(uv4 < (0.f - VECTOR_EPSILON)) || (uv4 >(1.f + VECTOR_EPSILON))) {
			idx = -1; // not valid
		}
		else {// valid
			isInside = true;

			return points[(*restInfo).currIndices[4 * idx + 0]] * uv4 +
				points[(*restInfo).currIndices[4 * idx + 1]] * uvw.x +
				points[(*restInfo).currIndices[4 * idx + 2]] * uvw.y +
				points[(*restInfo).currIndices[4 * idx + 3]] * uvw.z;
		}
	}

	
	Vec3 pWorld(0.);	
	
	idx = isInsideOpt((*restInfo).restTets, (*restInfo).defoInvRest, pPatch, Vec3(0.), Vec3(1.), &uvw) / 4;
	if (hintTetID) { *hintTetID = idx; }
	if (idx >= 0) {
		Real uv4 = (1. - uvw[0] - uvw[1] - uvw[2]);
		pWorld = points[(*restInfo).currIndices[4 * idx + 0]] * uv4 +
			points[(*restInfo).currIndices[4 * idx + 1]] * uvw.x +
			points[(*restInfo).currIndices[4 * idx + 2]] * uvw.y +
			points[(*restInfo).currIndices[4 * idx + 3]] * uvw.z;
		isInside = true;
	}
	else {
		isInside = false; // should be optimized by compiler
	}
	return pWorld;

}

Vec3 Patch_DefoData::world2Local(const Vec3& pWorld, bool* isInside /*= NULL*/, int * hintTetID /*= NULL*/) const
{
	int pTetID = -1;
	Vec3 uvw;
	Real uv4;
	if (isInside) *isInside = false;

	if (hintTetID != NULL && (*hintTetID) >= 0) { // validate hint
		pTetID = *hintTetID;
		Vec3  pTr = pWorld - tets[4 * pTetID];
		uvw = defoInv[pTetID] * pTr;
		uv4 = (1. - uvw[0] - uvw[1] - uvw[2]);

		if ((uvw[0]     < (0.f - VECTOR_EPSILON)) || (uvw[0]     > (1.f + VECTOR_EPSILON)) || (uvw[1] < (0.f - VECTOR_EPSILON)) ||
			(uvw[1]     > (1.f + VECTOR_EPSILON)) || (uvw[2]     < (0.f - VECTOR_EPSILON)) || (uvw[2] > (1.f + VECTOR_EPSILON)) ||
			(uv4 < (0.f - VECTOR_EPSILON)) || (uv4 >(1.f + VECTOR_EPSILON))) {
			pTetID = -1; // not valid
		} else {// valid
			if (isInside) *isInside = true;

			return (*restInfo).restPoints[(*restInfo).currIndices[4 * pTetID + 0]] * uv4 +
				(*restInfo).restPoints[(*restInfo).currIndices[4 * pTetID + 1]] * uvw.x +
				(*restInfo).restPoints[(*restInfo).currIndices[4 * pTetID + 2]] * uvw.y +
				(*restInfo).restPoints[(*restInfo).currIndices[4 * pTetID + 3]] * uvw.z;
		}
	}

	Vec3 pPatch(0.);
	pTetID = int(this->isInside(pWorld, &uvw) / 4.0); //isInsideOpt(points, defoInv, pWorld, bbmin, bbmax, &uvw);

	if (pTetID >= 0) {
		uv4 = (1. - uvw[0] - uvw[1] - uvw[2]);

		pPatch = (*restInfo).restPoints[(*restInfo).currIndices[4 * pTetID + 0]] * uv4 +
			(*restInfo).restPoints[(*restInfo).currIndices[4 * pTetID + 1]] * uvw.x +
			(*restInfo).restPoints[(*restInfo).currIndices[4 * pTetID + 2]] * uvw.y +
			(*restInfo).restPoints[(*restInfo).currIndices[4 * pTetID + 3]] * uvw.z;
		if (isInside) *isInside = true;
	}
	if (hintTetID != NULL)
		*hintTetID = pTetID;

	return pPatch;
}

Vec3 Patch_DefoData::localVec2WorldVec(const Vec3& pPatch, const Vec3& vPatch, int * hintTetID /*= NULL*/) const
{
	int tetID = -1;
	if (hintTetID) tetID = *hintTetID;
	Vec3 defoPos = local2World(pPatch, &tetID);
	if (hintTetID)*hintTetID = tetID;
	if (tetID < 0) return Vec3(0.0);

	Real j_step = 0.025f;
	int newtetID(tetID);
	Vec3 defoPosX1(defoPos), defoPosY1(defoPos), defoPosZ1(defoPos), \
		defoPosX0(defoPos), defoPosY0(defoPos), defoPosZ0(defoPos);
	if (pPatch.x + j_step < 1.0f)
		defoPosX1 = local2World(pPatch + Vec3(j_step, 0.0f, 0.0f), &newtetID);
	newtetID=tetID;
	if (pPatch.y + j_step < 1.0f)
		defoPosY1 = local2World(pPatch + Vec3(0.0f, j_step, 0.0f), &newtetID);
	newtetID = tetID;
	if (pPatch.z + j_step < 1.0f)
		defoPosZ1 = local2World(pPatch + Vec3(0.0f, 0.0f, j_step), &newtetID);
	newtetID = tetID;
	if (pPatch.x - j_step > 0.0f)
		defoPosX0 = local2World(pPatch - Vec3(j_step, 0.0f, 0.0f), &newtetID);
	newtetID = tetID;
	if (pPatch.y - j_step > 0.0f)
		defoPosY0 = local2World(pPatch - Vec3(0.0f, j_step, 0.0f), &newtetID);
	newtetID = tetID;
	if (pPatch.z - j_step > 0.0f)
		defoPosZ0 = local2World(pPatch - Vec3(0.0f, 0.0f, j_step), &newtetID);

	j_step *= 2.0f;
	Matrix4x4<Real> J_deform((defoPosX1 - defoPosX0) / j_step, (defoPosY1 - defoPosY0) / j_step, (defoPosZ1 - defoPosZ0) / j_step);
	Vec3 defoDir = J_deform * vPatch; // 2D: cubeDir.z is already 0.0 in this way
	// rescale
	normalize(defoDir);
	defoDir *= norm(vPatch);
	return defoDir;
}

Vec3 Patch_DefoData::worldVec2LocalVec(const Vec3& pPatch, const Vec3& vWorld, int * hintTetID /*= NULL*/) const
{
	int tetID = -1;
	if (hintTetID) tetID = *hintTetID;
	Vec3 defoPos = local2World(pPatch, &tetID);
	if (hintTetID)*hintTetID = tetID;
	if (tetID < 0) return Vec3(0.0);

	bool isInside = false;
	Real j_step = 0.0002f;
	int newtetID(tetID);

	Vec3 cubePosX1 = world2Local(defoPos + Vec3(j_step, 0.0f, 0.0f), &isInside, &newtetID);
	if (!isInside) cubePosX1 = pPatch;
	newtetID = tetID;
	Vec3 cubePosY1 = world2Local(defoPos + Vec3(0.0f, j_step, 0.0f), &isInside, &newtetID);
	if (!isInside) cubePosY1 = pPatch;
	newtetID = tetID;
	Vec3 cubePosZ1 = world2Local(defoPos + Vec3(0.0f, 0.0f, j_step), &isInside, &newtetID);
	if (!isInside) cubePosZ1 = pPatch;
	newtetID = tetID;
	Vec3 cubePosX0 = world2Local(defoPos - Vec3(j_step, 0.0f, 0.0f), &isInside, &newtetID);
	if (!isInside) cubePosX0 = pPatch;
	newtetID = tetID;
	Vec3 cubePosY0 = world2Local(defoPos - Vec3(0.0f, j_step, 0.0f), &isInside, &newtetID);
	if (!isInside) cubePosY0 = pPatch;
	newtetID = tetID;
	Vec3 cubePosZ0 = world2Local(defoPos - Vec3(0.0f, 0.0f, j_step), &isInside, &newtetID);
	if (!isInside) cubePosZ0 = pPatch;
	j_step *= 2.0f;
	Matrix4x4<Real> J_undeform((cubePosX1 - cubePosX0) / j_step, (cubePosY1 - cubePosY0) / j_step, (cubePosZ1 - cubePosZ0) / j_step);
	Vec3 cubeDir = J_undeform * vWorld; // 2D: cubeDir.z is already 0.0 in this way
	// rescale
	normalize(cubeDir);
	cubeDir *= norm(vWorld);
	return cubeDir;
}

// need to rebuild according to current vertex positions
void GMatrixGen(int subd, const std::vector<Vec3> & vertex, SparseMatrix<Real>& GS) {
	// assert size matches!
	int pSize = vertex.size();
	int m = 3 * pSize;

	GS.setZero();
	GS.reserve(VectorXi::Constant(m, 10)); // each row has at least 10 entries, at most 19 entries

	const int n = 2 + subd;
	int p[8] = { 0 };
	//VectorXf curVn(m);//debug
	for (int cubk = 0; cubk < subd + 1; ++cubk)
	for (int cubj = 0; cubj < subd + 1; ++cubj)
	for (int cubi = 0; cubi < subd + 1; ++cubi) {// go through all cube
												 // current cube vertex, 0 - (pSize - 1)	
		p[0] = cubk*n*n + cubj*n + cubi;
		p[1] = p[0] + 1; p[2] = p[0] + n;
		p[3] = p[2] + 1;
		p[4] = p[0] + n*n;
		p[5] = p[4] + 1; p[6] = p[4] + n;
		p[7] = p[6] + 1;

		for (int vv3 = 0; vv3 < 8; ++vv3) { // for every vertex in the cube
			int v[4];// neib vertex, 0 - (pSize - 1)
			v[0] = p[neibIdx[vv3 * 3 + 0]];
			v[1] = p[neibIdx[vv3 * 3 + 1]];
			v[2] = p[neibIdx[vv3 * 3 + 2]];
			v[3] = p[vv3];

			Vec3 n02 = vertex[v[2]] - vertex[v[0]]; normalize(n02);
			Vec3 n10 = vertex[v[0]] - vertex[v[1]]; normalize(n10);
			Vec3 n21 = vertex[v[1]] - vertex[v[2]]; normalize(n21);

			Matrix3f R02 = AngleAxisf(0.5*M_PI, Vector3f(n02.x, n02.y, n02.z)).toRotationMatrix();
			Matrix3f R10 = AngleAxisf(0.5*M_PI, Vector3f(n10.x, n10.y, n10.z)).toRotationMatrix();
			Matrix3f R21 = AngleAxisf(0.5*M_PI, Vector3f(n21.x, n21.y, n21.z)).toRotationMatrix();

			Matrix3f A[4];
			Real wei1 = sqrtf(2.0f) / 3.0f, wei2 = 3.0f;
			A[0] = (Matrix3f::Identity() + (R02*0.5 + R10*0.5 - R21)*wei1) / wei2;
			A[1] = (Matrix3f::Identity() + (R10*0.5 + R21*0.5 - R02)*wei1) / wei2;
			A[2] = (Matrix3f::Identity() + (R21*0.5 + R02*0.5 - R10)*wei1) / wei2;
			A[3] = -Matrix3f::Identity();


			for (int ai = 0; ai < 4; ++ai)// fill in neib weight
			for (int aj = ai; aj < 4; ++aj) { // started from ai! skip half, get filled by symmetric

				Matrix3f AIJ = A[ai].transpose() * A[aj];

				//cout << "a" << ai << "^T*a" << aj << "is:\n" << AIJ << endl;

				int gi = v[ai];
				int gj = v[aj];

				if (gi > gj) { // symmetric, only fill half
					gi = v[aj];
					gj = v[ai];
					AIJ.transposeInPlace();
				}

				for (int bi = 0; bi < 3; ++bi)
				for (int bj = 0; bj < 3; ++bj) {// symmetric
					int ti = gi * 3 + bi;
					int tj = gj * 3 + bj;
					if (ti == tj)
						GS.coeffRef(ti, tj) += AIJ(bi, bj);
					else if (ti < tj) {
						GS.coeffRef(ti, tj) += AIJ(bi, bj);
						GS.coeffRef(tj, ti) += AIJ(bi, bj);
					}
				}

			}
		}
	}
	GS.makeCompressed();// optional
}

bool GSolverGen(int subd, Real lambda, const std::vector<Vec3> & vertex,
	SimplicialCholesky< SparseMatrix<Real> >& solver, std::vector<Vec3> & newvertex, 
	float* curError = NULL) {

	int pSize = vertex.size();
	int m = 3 * pSize;
	SparseMatrix<Real> GS(m, m);
	GMatrixGen(subd, vertex, GS);// get the G grid

	if (curError) {
		*curError = 0.0f;
		//float curScale = 0.0f;
		for (int ek = 0; ek < GS.outerSize(); ++ek)
		for (SparseMatrix<Real>::InnerIterator it(GS, ek); it; ++it) {
			(*curError) += it.value() * vertex[it.row() / 3][it.row() % 3]
				* vertex[it.col() / 3][it.col() % 3];
		}
	}

	SparseMatrix<Real> GI(m, m);
	GI.setIdentity();

	GI = GS * lambda + GI;
	GI.makeCompressed();// optional

	solver.compute(GI);

	if (solver.info() != Eigen::Success) {
		debMsg("not invertable!!", 0);
		if (curError) {
			*curError = 0.0f;
		}
		return false;
	}

	VectorXf xyz(pSize * 3), Vn(pSize * 3);
	for (int wi = 0; wi < pSize; ++wi) {
		Vn(3 * wi + 0) = vertex[wi].x;
		Vn(3 * wi + 1) = vertex[wi].y;
		Vn(3 * wi + 2) = vertex[wi].z;
	}
	xyz = solver.solve(Vn);
	if (curError) {
		*curError = xyz.transpose() * GS * xyz;
		//*curError = error(0);
	}
	for (int wi = 0; wi < pSize; ++wi) {
		newvertex[wi].x = xyz[wi * 3 + 0];
		newvertex[wi].y = xyz[wi * 3 + 1];
		newvertex[wi].z = xyz[wi * 3 + 2];
	}	
	return true;
}

Real adjustDefo(int subd, const std::vector<Vec3> & vertex, std::vector<Vec3> & newvertex, Real lambda = 0.0f) {

	//int subd = restInfo->subdN;
	Real curLambda = lambda * (subd + 2.0f)* (subd + 2.0f)* (subd + 2.0f) / (subd + 1.0f);
	Real curErr = 0.0f;
	int pSize = vertex.size();
	
	SimplicialCholesky< SparseMatrix<Real> > solver;
	GSolverGen(subd, curLambda, vertex, solver, newvertex, &curErr);
	return curErr;
}

void FVectorGen(const int subd, const std::vector<Vec3> & Tvertex, const int index, VectorXf& arrayF) {
	// assert size matches!
	// arrayF.size() = m;
	// Tvertex.size() = 8*t;
	int n = 2 + subd;
	int t = (1 + subd)*(1 + subd)*(1 + subd);
	int m = (2 + subd)*(2 + subd)*(2 + subd);

	arrayF.setZero();	
	int p[8] = { 0 };
	
	for (int cubk = 0; cubk < subd + 1; ++cubk)
	for (int cubj = 0; cubj < subd + 1; ++cubj)
	for (int cubi = 0; cubi < subd + 1; ++cubi) {// go through all cube
											 // current cube vertex, 0 - (pSize - 1)	
		p[0] = cubk*n*n + cubj*n + cubi;
		p[1] = p[0] + 1; p[2] = p[0] + n;
		p[3] = p[2] + 1;
		p[4] = p[0] + n*n;
		p[5] = p[4] + 1; p[6] = p[4] + n;
		p[7] = p[6] + 1;

		for (int eij = 0; eij < 12; ++eij) { // for every edge in the cube
			int v[2] = { p[edgeIdx[eij * 2 + 0]] , p[edgeIdx[eij * 2 + 1]] };
			int wCube = cubk* (1 + subd)*(1 + subd) + cubj*(1 + subd) + cubi;
			int w[2] = { (wCube)*8 + edgeIdx[eij * 2 + 0], (wCube)* 8 + edgeIdx[eij * 2 + 1] };
			// 2.0, Gen -F/2 directly instead of F it self, since v = H ^(-1) * (-f/2 )
			arrayF(v[0]) += (Tvertex[w[0]][index] - Tvertex[w[1]][index]);
			arrayF(v[1]) += (Tvertex[w[1]][index] - Tvertex[w[0]][index]);
		}
	}
}

void TvertexGen(const int subd, const std::vector<Vec3> & vertex, const Real patchLen, std::vector<Vec3> & Tvertex) {
	// assert size matches!
	// Tvertex.size() = 8*t;
	int n = 2 + subd;
	int t = (1 + subd)*(1 + subd)*(1 + subd);
	int m = (2 + subd)*(2 + subd)*(2 + subd);

	Tvertex.clear();
	Tvertex.resize(8 * t);

	int p[8] = { 0 };

	for (int cubk = 0; cubk < subd + 1; ++cubk)
	for (int cubj = 0; cubj < subd + 1; ++cubj)
	for (int cubi = 0; cubi < subd + 1; ++cubi){// go through all cube
		// current cube index, 0 - (m - 1), in array vertex
		p[0] = cubk*n*n + cubj*n + cubi;
		p[1] = p[0] + 1; p[2] = p[0] + n;
		p[3] = p[2] + 1;
		p[4] = p[0] + n*n;
		p[5] = p[4] + 1; p[6] = p[4] + n;
		p[7] = p[6] + 1;

		// index in array Tvertex is, wCube * 8 + (0...7)
		int wCube = cubk* (1 + subd)*(1 + subd) + cubj*(1 + subd) + cubi;
		Vec4 FRb2[4] = {
			Vec4(1.0,0.5,0.5,0.5),
			Vec4(0.5,1.0,0.0,0.0),
			Vec4(0.5,0.0,1.0,0.0),
			Vec4(0.5,0.0,0.0,1.0)
		};
		Vec3 wT[4];
		for (int index = 0; index < 3; ++index) {
			Vec4 NCd2(
				(vertex[p[0]][index] - vertex[p[3]][index] - vertex[p[5]][index] - vertex[p[6]][index] - vertex[p[7]][index] * 2.0),
				(vertex[p[1]][index] + vertex[p[3]][index] + vertex[p[5]][index] + vertex[p[7]][index]),
				(vertex[p[2]][index] + vertex[p[3]][index] + vertex[p[6]][index] + vertex[p[7]][index]),
				(vertex[p[4]][index] + vertex[p[5]][index] + vertex[p[6]][index] + vertex[p[7]][index]));
			for (int wiii = 0; wiii < 4; ++wiii)
				wT[wiii][index] = dot(FRb2[wiii], NCd2);
		}
		for (int wiii = 1; wiii < 4; ++wiii) {
			Vec3 d = getNormalized(wT[wiii] - wT[0]);
			wT[wiii] = wT[0] + ( patchLen / (1.0 + subd) )*d;
		}
		Tvertex[wCube * 8 + 0] = wT[0];
		Tvertex[wCube * 8 + 1] = wT[1];
		Tvertex[wCube * 8 + 2] = wT[2];
		Tvertex[wCube * 8 + 3] = wT[1] + wT[2] - wT[0];
		Tvertex[wCube * 8 + 4] = wT[3];
		Tvertex[wCube * 8 + 5] = wT[1] + wT[3] - wT[0];
		Tvertex[wCube * 8 + 6] = wT[2] + wT[3] - wT[0];
		Tvertex[wCube * 8 + 7] = wT[1] + wT[2] + wT[3] - wT[0] * 2.0;
	}
}

void Patch_DefoData::cageAvgBase(Vec3& dirX, Vec3& dirY, Vec3& dirZ) {
	dirX = dirY = dirZ = Vec3(0.0);
	Real dirXW(0.0), dirYW(0.0), dirZW(0.0);
	Vec3 curpos(0.0); Real curWei(0.0);
	const int subd = restInfo->subdN;
	const int n = 2 + subd;

	int vn = points.size();
	for (int vi = 0; vi < vn; ++vi) {
		if ((vi%n) < (n - 1)) {
			curpos = (restInfo->restPoints[vi] + restInfo->restPoints[vi + 1])*Real(0.5);
			curWei = getConeWeight(/*2.0f**/norm(curpos - Vec3(.5)));
			dirX += (points[vi + 1] - points[vi])* curWei;
			dirXW += curWei;
		}
		if (((vi / n) % n) < (n - 1)) {
			curpos = (restInfo->restPoints[vi] + restInfo->restPoints[vi + n])*Real(0.5);
			curWei = getConeWeight(/*2.0f**/norm(curpos - Vec3(.5)));
			dirY += (points[vi + n] - points[vi])* curWei;
			dirYW += curWei;
		}
		if (((vi / n) / n) < (n - 1)) {
			curpos = (restInfo->restPoints[vi] + restInfo->restPoints[vi + n*n])*Real(0.5);
			curWei = getConeWeight(/*2.0f**/norm(curpos - Vec3(.5)));
			dirZ += (points[vi + n*n] - points[vi] )* curWei;
			dirZW += curWei;
		}
	}

	dirX /= dirXW; dirY /= dirYW; dirZ /= dirZW;
}

void Patch_DefoData::cageAvgBaseLen(Real& dirXL, Real& dirYL, Real& dirZL){
	dirXL = dirYL = dirZL = (0.0);
	Real dirXW(0.0), dirYW(0.0), dirZW(0.0);
	Vec3 curpos(0.0); Real curWei(0.0);
	const int subd = restInfo->subdN;
	const int n = 2 + subd;

	int vn = points.size();
	for (int vi = 0; vi < vn; ++vi) {
		if ((vi%n) < (n - 1)) {
			curpos = (restInfo->restPoints[vi] + restInfo->restPoints[vi + 1])*Real(0.5);
			curWei = getConeWeight(/*2.0f**/norm(curpos - Vec3(.5)));
			dirXL += norm(points[vi + 1] - points[vi])* curWei;
			dirXW += curWei;
		}
		if (((vi / n) % n) < (n - 1)) {
			curpos = (restInfo->restPoints[vi] + restInfo->restPoints[vi + n])*Real(0.5);
			curWei = getConeWeight(/*2.0f**/norm(curpos - Vec3(.5)));
			dirYL += norm(points[vi + n] - points[vi])* curWei;
			dirYW += curWei;
		}
		if (((vi / n) / n) < (n - 1)) {
			curpos = (restInfo->restPoints[vi] + restInfo->restPoints[vi + n*n])*Real(0.5);
			curWei = getConeWeight(/*2.0f**/norm(curpos - Vec3(.5)));
			dirZL += norm(points[vi + n*n] - points[vi])* curWei;
			dirZW += curWei;
		}
	}

	dirXL /= dirXW; dirYL /= dirYW; dirZL /= dirZW;
}

Real Patch_DefoData::adjustDefoInPlace(std::vector<Vec3> & vertex, Real lambda) {
	Real adError = adjustDefo(restInfo->subdN, vertex, points, lambda);
	//init(*restInfo);
	return adError;
}

void Patch_DefoData::adjustScaleInPlace(Real patchLen, Vec3& oldC)
{
	SimplicialCholesky< SparseMatrix<Real> >* solver = (SimplicialCholesky< SparseMatrix<Real> >*) (restInfo->HS);
	std::vector<Vec3> newP(points);
	
	if (solver) {
		int m = points.size(); //  m = (subdN+2)^3
		VectorXf X(m), FV(m);
		std::vector<Vec3> TW;
		Real centreWei = 0.0;
		Vec3 centreP(0.0);
		for (size_t i = 0; i < 3; i++)
		{
			TvertexGen(restInfo->subdN, points, patchLen, TW);
			FVectorGen(restInfo->subdN, TW, i, FV);
			X = solver->solve(FV);

			for (int wi = 0; wi < m; ++wi) {
				newP[wi][i] = X(wi);
				Vec3 curpos = restInfo->restPoints[wi];
				Real curWei = getConeWeight(/*2.0f**/norm(curpos - Vec3(.5)));
				// do not want 0 for boundary points, always positive wei
				if (i == 0) centreWei += curWei;
				centreP[i] += X(wi) * curWei;
			}
		}
		centreP /= centreWei;
		for (int wi = 0; wi < m; ++wi) {
			points[wi] = newP[wi] - centreP + oldC;
		}
	}
}

void Patch_DefoData::meshView(Mesh& mesh, Vec3 factor)
{
	int vn = points.size();
	// if (is2D) vn /= (2 + subd); // only 3D has mesh

	int p[8] = { 0 };
	const int n = 2 + restInfo->subdN;
	//VectorXf curVn(m);//debug
	//int nck = is2D ? 1 : (n - 1);
	for (int cubk = 0; cubk < n - 1; ++cubk)
	for (int cubj = 0; cubj < n - 1; ++cubj)
	for (int cubi = 0; cubi < n - 1; ++cubi) {// go through all cube
		p[0] = cubk*n*n + cubj*n + cubi;
		p[1] = p[0] + 1; p[2] = p[0] + n;
		p[3] = p[2] + 1;
		// if 3D
		p[4] = p[0] + n*n;
		p[5] = p[4] + 1; p[6] = p[4] + n;
		p[7] = p[6] + 1;

		Node vertice[8];
		int meshNodeSize = mesh.numNodes();
		for (int fvi = 0; fvi < 8; ++fvi) {
			vertice[fvi].pos = points[ p[fvi] ] * factor;
			mesh.addNode(vertice[fvi]);
		}
		//Vec3 normal = vertice[fvi].pos
		if (cubk == 0) {
			mesh.addTri(Triangle(meshNodeSize + 0, meshNodeSize + 1, meshNodeSize + 2));
			mesh.addTri(Triangle(meshNodeSize + 1, meshNodeSize + 3, meshNodeSize + 2));

			mesh.addTri(Triangle(meshNodeSize + 0, meshNodeSize + 2, meshNodeSize + 1));
			mesh.addTri(Triangle(meshNodeSize + 1, meshNodeSize + 2, meshNodeSize + 3));
		}

		if (cubi == 0) {
			mesh.addTri(Triangle(meshNodeSize + 2, meshNodeSize + 0, meshNodeSize + 4));
			mesh.addTri(Triangle(meshNodeSize + 2, meshNodeSize + 4, meshNodeSize + 6));

			mesh.addTri(Triangle(meshNodeSize + 2, meshNodeSize + 4, meshNodeSize + 0));
			mesh.addTri(Triangle(meshNodeSize + 2, meshNodeSize + 6, meshNodeSize + 4));
		}

		if (cubj == 0) {
			mesh.addTri(Triangle(meshNodeSize + 0, meshNodeSize + 1, meshNodeSize + 4));
			mesh.addTri(Triangle(meshNodeSize + 1, meshNodeSize + 5, meshNodeSize + 4));

			mesh.addTri(Triangle(meshNodeSize + 0, meshNodeSize + 4, meshNodeSize + 1));
			mesh.addTri(Triangle(meshNodeSize + 1, meshNodeSize + 4, meshNodeSize + 5));
		}

		mesh.addTri(Triangle(meshNodeSize + 2, meshNodeSize + 3, meshNodeSize + 6));
		mesh.addTri(Triangle(meshNodeSize + 3, meshNodeSize + 7, meshNodeSize + 6));

		mesh.addTri(Triangle(meshNodeSize + 4, meshNodeSize + 5, meshNodeSize + 6));
		mesh.addTri(Triangle(meshNodeSize + 6, meshNodeSize + 5, meshNodeSize + 7));

		mesh.addTri(Triangle(meshNodeSize + 1, meshNodeSize + 5, meshNodeSize + 3));
		mesh.addTri(Triangle(meshNodeSize + 3, meshNodeSize + 5, meshNodeSize + 7));

		// double sided
		mesh.addTri(Triangle(meshNodeSize + 4, meshNodeSize + 6, meshNodeSize + 5));
		mesh.addTri(Triangle(meshNodeSize + 6, meshNodeSize + 7, meshNodeSize + 5));

		mesh.addTri(Triangle(meshNodeSize + 1, meshNodeSize + 3, meshNodeSize + 5));
		mesh.addTri(Triangle(meshNodeSize + 3, meshNodeSize + 7, meshNodeSize + 5));

		mesh.addTri(Triangle(meshNodeSize + 2, meshNodeSize + 6, meshNodeSize + 3));
		mesh.addTri(Triangle(meshNodeSize + 3, meshNodeSize + 6, meshNodeSize + 7));
	}

}

}