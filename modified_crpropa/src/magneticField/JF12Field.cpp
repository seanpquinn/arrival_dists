#include "crpropa/magneticField/JF12Field.h"
#include "crpropa/Units.h"
#include "crpropa/GridTools.h"
#include "crpropa/Random.h"

#include <iostream>

namespace crpropa {

double logisticFunction(double x, double x0, double w) {
	return 1. / (1. + exp(-2. * (fabs(x) - x0) / w));
}

JF12Field::JF12Field(int seed, double rs, double wh, double r0Turb, double z0Turb) {
	useRegular = true;
	useStriated = false;
	useTurbulent = false;

  Random regRand;
  regRand.seed(seed);

  // This model has been modified to use updated parameters
  // published by the Planck collaboration in
  // arxiv.org/abs/1601.00546v2
  // These updates should be considered more robust than
  // original values due to inclusion of Planck dust maps

	// spiral arm parameters
  pitch = 11.5 * M_PI / 180;
	sinPitch = sin(pitch);
	cosPitch = cos(pitch);
	tan90MinusPitch = tan(M_PI / 2 - pitch);
  
  // r_x
	rArms[0] = 4.947 * kpc;
	rArms[1] = 6.111 * kpc;
	rArms[2] = 6.887 * kpc;
	rArms[3] = 8.051 * kpc;
	rArms[4] = 9.506 * kpc;
	rArms[5] = 11.058 * kpc;
	rArms[6] = 12.319 * kpc;
	rArms[7] = 15.035 * kpc;

	// regular field parameters

  //   --- disk parameters
  bRing = regRand.randNorm(0.1,0.01) * muG;
  hDisk = regRand.randNorm(0.40,0.0009) * kpc;
	wDisk = regRand.randNorm(0.27,0.0064) * kpc;

	bDisk[0] = regRand.randNorm(0.1,3.24) * muG;  // b1
	bDisk[1] = regRand.randNorm(2.4,.36) * muG;   // b2
	bDisk[2] = regRand.randNorm(-0.9,.64) * muG;  // b3
	bDisk[3] = regRand.randNorm(0.88,.09) * muG;  // b4
	bDisk[4] = regRand.randNorm(-2.6,.01) * muG;  // b5
	bDisk[5] = regRand.randNorm(-3.78,.25) * muG; // b6
	bDisk[6] = regRand.randNorm(0.0,3.24) * muG;  // b7
	bDisk[7] = regRand.randNorm(2.7,3.24) * muG;  // b8

  //   --- toroidal halo parameters
	bNorth = regRand.randNorm(1.16,.01) * muG;
	bSouth = regRand.randNorm(-0.92,.01) * muG;
	rNorth = regRand.randNorm(9.22,.0064) * kpc;
	rSouth = rs * kpc;
	wHalo = wh * kpc;
	z0 = regRand.randNorm(5.3,2.56) * kpc;

  //   --- X halo parameters
	bX = regRand.randNorm(3.64,.09) * muG;
	thetaX0 = regRand.randNorm(49.0,1.) * M_PI / 180;
	sinThetaX0 = sin(thetaX0);
	cosThetaX0 = cos(thetaX0);
	tanThetaX0 = tan(thetaX0);
	rXc = regRand.randNorm(4.8,.04) * kpc;
	rX = regRand.randNorm(2.9,.01) * kpc;

	// striated field parameter
	sqrtbeta = regRand.randNorm(sqrt(6.54),0.1296);

	// turbulent field parameters
	bDiskTurb[0] = regRand.randNorm(4.972,5.4289) * muG;  // b1
	bDiskTurb[1] = regRand.randNorm(6.126,2.4964) * muG;  // b2
	bDiskTurb[2] = regRand.randNorm(4.412,1.21) * muG;    // b3
	bDiskTurb[3] = regRand.randNorm(6.126,0.7569) * muG;  // b4
	bDiskTurb[4] = regRand.randNorm(0.904,1.7424) * muG;  // b5
	bDiskTurb[5] = regRand.randNorm(22.22,6.401) * muG;   // b6
	bDiskTurb[6] = regRand.randNorm(17.154,5.7121) * muG; // b7
	bDiskTurb[7] = regRand.randNorm(9.11,19.625) * muG;   // b8

	bDiskTurb5 = regRand.randNorm(5.39,1.932) * muG;      // bint
	zDiskTurb = regRand.randNorm(0.61,0.0016) * kpc;

	bHaloTurb = regRand.randNorm(4.68,1.9321) * muG;
	rHaloTurb = r0Turb * kpc;
	zHaloTurb = z0Turb * kpc;
}

void JF12Field::randomStriated(int seed) {
	useStriated = true;
	int N = 200;
	striatedGrid = new ScalarGrid(Vector3d(0.), N, 0.05 * kpc);

	Random random;
	if (seed != 0)
		random.seed(seed);

	for (int ix = 0; ix < N; ix++)
		for (int iy = 0; iy < N; iy++)
			for (int iz = 0; iz < N; iz++) {
				float &f = striatedGrid->get(ix, iy, iz);
				f = round(random.rand()) * 2 - 1;
			}
}

#ifdef CRPROPA_HAVE_FFTW3F
void JF12Field::randomTurbulent(int seed) {
	useTurbulent = true;
	// turbulent field with Kolmogorov spectrum, B_rms = 1 and Lc = 60 parsec
	turbulentGrid = new VectorGrid(Vector3d(0.), 512, 2 * parsec);
	initTurbulence(turbulentGrid, 7.8, 10 * parsec, 224 * parsec, -11./3., seed);
}
#endif

void JF12Field::setStriatedGrid(ref_ptr<ScalarGrid> grid) {
	useStriated = true;
	striatedGrid = grid;
}

void JF12Field::setTurbulentGrid(ref_ptr<VectorGrid> grid) {
	useTurbulent = true;
	turbulentGrid = grid;
}

ref_ptr<ScalarGrid> JF12Field::getStriatedGrid() {
	return striatedGrid;
}

ref_ptr<VectorGrid> JF12Field::getTurbulentGrid() {
	return turbulentGrid;
}

void JF12Field::setUseRegular(bool use) {
	useRegular = use;
}

void JF12Field::setUseStriated(bool use) {
	if ((use) and (striatedGrid)) {
		std::cout << "JF12Field: No striated field set: ignored" << std::endl;
		return;
	}
	useStriated = use;
}

void JF12Field::setUseTurbulent(bool use) {
	if ((use) and (turbulentGrid)) {
		std::cout << "JF12Field: No turbulent field set: ignored" << std::endl;
		return;
	}
	useTurbulent = use;
}

bool JF12Field::isUsingRegular() {
	return useRegular;
}

bool JF12Field::isUsingStriated() {
	return useStriated;
}

bool JF12Field::isUsingTurbulent() {
	return useTurbulent;
}

Vector3d JF12Field::getRegularField(const Vector3d& pos) const {
	Vector3d b(0.);

	double r = sqrt(pos.x * pos.x + pos.y * pos.y); // in-plane radius
	double d = pos.getR(); // distance to galactic center
	if ((d < 1 * kpc) or (d > 20 * kpc))
		return b; // 0 field for d < 1 kpc or d > 20 kpc

	double phi = pos.getPhi(); // azimuth
	double sinPhi = sin(phi);
	double cosPhi = cos(phi);

	double lfDisk = logisticFunction(pos.z, hDisk, wDisk);

	// disk field
	if (r > 3 * kpc) {
		double bMag;
		if (r < 5 * kpc) {
			// molecular ring
			bMag = bRing * (5 * kpc / r) * (1 - lfDisk);
			b.x += -bMag * sinPhi;
			b.y += bMag * cosPhi;

		} else {
			// spiral region
			double r_negx = r * exp(-(phi - M_PI) / tan90MinusPitch);
			if (r_negx > rArms[7])
				r_negx = r * exp(-(phi + M_PI) / tan90MinusPitch);
			if (r_negx > rArms[7])
				r_negx = r * exp(-(phi + 3 * M_PI) / tan90MinusPitch);

			for (int i = 7; i >= 0; i--)
				if (r_negx < rArms[i])
					bMag = bDisk[i];

			bMag *= (5 * kpc / r) * (1 - lfDisk);
			b.x += bMag * (sinPitch * cosPhi - cosPitch * sinPhi);
			b.y += bMag * (sinPitch * sinPhi + cosPitch * cosPhi);
		}
	}

	// toroidal halo field
	double bMagH = exp(-fabs(pos.z) / z0) * lfDisk;
	if (pos.z >= 0)
		bMagH *= bNorth * (1 - logisticFunction(r, rNorth, wHalo));
	else
		bMagH *= bSouth * (1 - logisticFunction(r, rSouth, wHalo));
	b.x += -bMagH * sinPhi;
	b.y += bMagH * cosPhi;

	// poloidal halo field
	double bMagX;
	double sinThetaX, cosThetaX;
	double rp;
	double rc = rXc + fabs(pos.z) / tanThetaX0;
	if (r < rc) {
		// varying elevation region
		rp = r * rXc / rc;
		bMagX = bX * exp(-1 * rp / rX) * pow(rXc / rc, 2.);
		double thetaX = atan2(fabs(pos.z), (r - rp));
		if (pos.z == 0)
			thetaX = M_PI / 2.;
		sinThetaX = sin(thetaX);
		cosThetaX = cos(thetaX);
	} else {
		// constant elevation region
		rp = r - fabs(pos.z) / tanThetaX0;
		bMagX = bX * exp(-rp / rX) * (rp / r);
		sinThetaX = sinThetaX0;
		cosThetaX = cosThetaX0;
	}
	double zsign = pos.z < 0 ? -1 : 1;
	b.x += zsign * bMagX * cosThetaX * cosPhi;
	b.y += zsign * bMagX * cosThetaX * sinPhi;
	b.z += bMagX * sinThetaX;

	return b;
}

Vector3d JF12Field::getStriatedField(const Vector3d& pos) const {
	return (getRegularField(pos)
			* (1. + sqrtbeta * striatedGrid->closestValue(pos)));
}

double JF12Field::getTurbulentStrength(const Vector3d& pos) const {
	if (pos.getR() > 20 * kpc)
		return 0;

	double r = sqrt(pos.x * pos.x + pos.y * pos.y); // in-plane radius
	double phi = pos.getPhi(); // azimuth

	// disk
	double bDisk = 0;
	if (r < 5 * kpc) {
		bDisk = bDiskTurb5;
	} else {
		// spiral region
		double r_negx = r * exp(-(phi - M_PI) / tan90MinusPitch);
		if (r_negx > rArms[7])
			r_negx = r * exp(-(phi + M_PI) / tan90MinusPitch);
		if (r_negx > rArms[7])
			r_negx = r * exp(-(phi + 3 * M_PI) / tan90MinusPitch);

		for (int i = 7; i >= 0; i--)
			if (r_negx < rArms[i])
				bDisk = bDiskTurb[i];

		bDisk *= (5 * kpc) / r;
	}
	bDisk *= exp(-0.5 * pow(pos.z / zDiskTurb, 2));

	// halo
	double bHalo = bHaloTurb * exp(-r / rHaloTurb)
			* exp(-0.5 * pow(pos.z / zHaloTurb, 2));

	// modulate turbulent field
	return sqrt(pow(bDisk, 2) + pow(bHalo, 2));
}

Vector3d JF12Field::getTurbulentField(const Vector3d& pos) const {
	return (turbulentGrid->interpolate(pos) * getTurbulentStrength(pos));
}

Vector3d JF12Field::getField(const Vector3d& pos) const {
	Vector3d b(0.);
	if (useTurbulent)
		b += getTurbulentField(pos);
	if (useStriated)
		b += getStriatedField(pos);
	else if (useRegular)
		b += getRegularField(pos);
	return b;
}

} // namespace crpropa
