#include "crpropa/module/PhotoDisintegration.h"
#include "crpropa/Units.h"
#include "crpropa/ParticleID.h"
#include "crpropa/ParticleMass.h"
#include "crpropa/Random.h"

#include <cmath>
#include <limits>
#include <sstream>
#include <fstream>
#include <stdexcept>

namespace crpropa {

const double PhotoDisintegration::lgmin = 6;  // minimum log10(Lorentz-factor)
const double PhotoDisintegration::lgmax = 14; // maximum log10(Lorentz-factor)
const size_t PhotoDisintegration::nlg = 201;  // number of Lorentz-factor steps

PhotoDisintegration::PhotoDisintegration(PhotonField f, bool havePhotons, double limit) {
	setPhotonField(f);
	this->havePhotons = havePhotons;
	this->limit = limit;
}

void PhotoDisintegration::setPhotonField(PhotonField photonField) {
	this->photonField = photonField;
	switch (photonField) {
	case CMB:
		setDescription("PhotoDisintegration: CMB");
		initRate(getDataPath("pd_CMB.txt"));
		initBranching(getDataPath("pd_branching_CMB.txt"));
		initPhotonEmission(getDataPath("pd_photon_emission_CMB.txt"));
		break;
	case IRB:  // default: Kneiske '04 IRB model
	case IRB_Kneiske04:
		setDescription("PhotoDisintegration: IRB (Kneiske 2004)");
		initRate(getDataPath("pd_IRB_Kneiske04.txt"));
		initBranching(getDataPath("pd_branching_IRB_Kneiske04.txt"));
		initPhotonEmission(getDataPath("pd_photon_emission_IRB.txt"));
		break;
	case IRB_Stecker05:
		setDescription("PhotoDisintegration: IRB (Stecker 2005)");
		initRate(getDataPath("pd_IRB_Stecker05.txt"));
		initBranching(getDataPath("pd_branching_IRB_Stecker05.txt"));
		initPhotonEmission(getDataPath("pd_photon_emission_IRB.txt"));
		break;
	case IRB_Franceschini08:
		setDescription("PhotoDisintegration: IRB (Franceschini 2008)");
		initRate(getDataPath("pd_IRB_Franceschini08.txt"));
		initBranching(getDataPath("pd_branching_IRB_Franceschini08.txt"));
		initPhotonEmission(getDataPath("pd_photon_emission_IRB.txt"));
		break;
	case IRB_Finke10:
		setDescription("PhotoDisintegration: IRB (Finke 2010)");
		initRate(getDataPath("pd_IRB_Finke10.txt"));
		initBranching(getDataPath("pd_branching_IRB_Finke10.txt"));
		initPhotonEmission(getDataPath("pd_photon_emission_IRB.txt"));
		break;
	case IRB_Dominguez11:
		setDescription("PhotoDisintegration: IRB (Dominguez 2011)");
		initRate(getDataPath("pd_IRB_Dominguez11.txt"));
		initBranching(getDataPath("pd_branching_IRB_Dominguez11.txt"));
		initPhotonEmission(getDataPath("pd_photon_emission_IRB.txt"));
		break;
	case IRB_Gilmore12:
		setDescription("PhotoDisintegration: IRB (Gilmore 12)");
		initRate(getDataPath("pd_IRB_Gilmore12.txt"));
		initBranching(getDataPath("pd_branching_IRB_Gilmore12.txt"));
		initPhotonEmission(getDataPath("pd_photon_emission_IRB.txt"));
		break;
	default:
		throw std::runtime_error("PhotoDisintegration: unknown photon background");
	}
}

void PhotoDisintegration::setHavePhotons(bool havePhotons) {
	this->havePhotons = havePhotons;
}

void PhotoDisintegration::setLimit(double limit) {
	this->limit = limit;
}

void PhotoDisintegration::initRate(std::string filename) {
	std::ifstream infile(filename.c_str());
	if (not infile.good())
		throw std::runtime_error("PhotoDisintegration: could not open file " + filename);

	// clear previously loaded interaction rates
	pdRate.clear();
	pdRate.resize(27 * 31);

	std::string line;
	while (std::getline(infile, line)) {
		if (line[0] == '#')
			continue;
		std::stringstream lineStream(line);

		int Z, N;
		lineStream >> Z;
		lineStream >> N;

		double r;
		for (size_t i = 0; i < nlg; i++) {
			lineStream >> r;
			pdRate[Z * 31 + N].push_back(r / Mpc);
		}
	}
	infile.close();
}

void PhotoDisintegration::initBranching(std::string filename) {
	std::ifstream infile(filename.c_str());
	if (not infile.good())
		throw std::runtime_error("PhotoDisintegration: could not open file " + filename);

	// clear previously loaded interaction rates
	pdBranch.clear();
	pdBranch.resize(27 * 31);

	std::string line;
	while (std::getline(infile, line)) {
		if (line[0] == '#')
			continue;

		std::stringstream lineStream(line);

		int Z, N;
		lineStream >> Z;
		lineStream >> N;

		Branch branch;
		lineStream >> branch.channel;

		double r;
		for (size_t i = 0; i < nlg; i++) {
			lineStream >> r;
			branch.branchingRatio.push_back(r);
		}

		pdBranch[Z * 31 + N].push_back(branch);
	}

	infile.close();
}

void PhotoDisintegration::initPhotonEmission(std::string filename) {
	std::ifstream infile(filename.c_str());
	if (not infile.good())
		throw std::runtime_error("PhotoDisintegration: could not open file " + filename);

	// clear previously loaded emission probabilities
	pdPhoton.clear();

	std::string line;
	while (std::getline(infile, line)) {
		if (line[0] == '#')
			continue;

		std::stringstream lineStream(line);

		int Z, N, Zd, Nd;
		lineStream >> Z;
		lineStream >> N;
		lineStream >> Zd;
		lineStream >> Nd;

		PhotonEmission em;
		lineStream >> em.energy;
		em.energy *= eV;

		double r;
		for (size_t i = 0; i < nlg; i++) {
			lineStream >> r;
			em.emissionProbability.push_back(r);
		}

		int key = Z * 1000000 + N * 10000 + Zd * 100 + Nd;
		if (pdPhoton.find(key) == pdPhoton.end()) {
			std::vector<PhotonEmission> emissions;
			pdPhoton[key] = emissions;
		}
		pdPhoton[key].push_back(em);
	}

	infile.close();
}

void PhotoDisintegration::process(Candidate *candidate) const {
	// execute the loop at least once for limiting the next step
	double step = candidate->getCurrentStep();
	do {
		// check if nucleus
		int id = candidate->current.getId();
		if (not isNucleus(id))
			return;

		int A = massNumber(id);
		int Z = chargeNumber(id);
		int N = A - Z;
		size_t idx = Z * 31 + N;

		// check if disintegration data available
		if ((Z > 26) or (N > 30))
			return;
		if (pdRate[idx].size() == 0)
			return;

		// check if in tabulated energy range
		double z = candidate->getRedshift();
		double lg = log10(candidate->current.getLorentzFactor() * (1 + z));
		if ((lg <= lgmin) or (lg >= lgmax))
			return;

		double rate = interpolateEquidistant(lg, lgmin, lgmax, pdRate[idx]);
		rate *= pow(1 + z, 2) * photonFieldScaling(photonField, z); // cosmological scaling, rate per comoving distance

		// check if interaction occurs in this step
		// otherwise limit next step to a fraction of the mean free path
		Random &random = Random::instance();
		double randDist = -log(random.rand()) / rate;
		if (step < randDist) {
			candidate->limitNextStep(limit / rate);
			return;
		}

		// select channel and interact
		const std::vector<Branch> &branches = pdBranch[idx];
		double cmp = random.rand();
		int l = round((lg - lgmin) / (lgmax - lgmin) * (nlg - 1)); // index of closest tabulation point
		size_t i = 0;
		while ((i < branches.size()) and (cmp > 0)) {
			cmp -= branches[i].branchingRatio[l];
			i++;
		}
		performInteraction(candidate, branches[i-1].channel);

		// repeat with remaining step
		step -= randDist;
	} while (step > 0);
}

void PhotoDisintegration::performInteraction(Candidate *candidate, int channel) const {
	// parse disintegration channel
	int nNeutron = digit(channel, 100000);
	int nProton = digit(channel, 10000);
	int nH2 = digit(channel, 1000);
	int nH3 = digit(channel, 100);
	int nHe3 = digit(channel, 10);
	int nHe4 = digit(channel, 1);

	int dA = -nNeutron - nProton - 2 * nH2 - 3 * nH3 - 3 * nHe3 - 4 * nHe4;
	int dZ = -nProton - nH2 - nH3 - 2 * nHe3 - 2 * nHe4;

	int id = candidate->current.getId();
	int A = massNumber(id);
	int Z = chargeNumber(id);
	double EpA = candidate->current.getEnergy() / A;

	// update particle
	candidate->current.setId(nucleusId(A + dA, Z + dZ));
	candidate->current.setEnergy(EpA * (A + dA));

	// create secondaries
	Random &random = Random::instance();
	Vector3d pos = random.randomInterpolatedPosition(candidate->previous.getPosition(), candidate->current.getPosition());
	for (size_t i = 0; i < nNeutron; i++)
		candidate->addSecondary(nucleusId(1, 0), EpA, pos);
	for (size_t i = 0; i < nProton; i++)
		candidate->addSecondary(nucleusId(1, 1), EpA, pos);
	for (size_t i = 0; i < nH2; i++)
		candidate->addSecondary(nucleusId(2, 1), EpA * 2, pos);
	for (size_t i = 0; i < nH3; i++)
		candidate->addSecondary(nucleusId(3, 1), EpA * 3, pos);
	for (size_t i = 0; i < nHe3; i++)
		candidate->addSecondary(nucleusId(3, 2), EpA * 3, pos);
	for (size_t i = 0; i < nHe4; i++)
		candidate->addSecondary(nucleusId(4, 2), EpA * 4, pos);

	if (not havePhotons)
		return;

	// create photons
	double z = candidate->getRedshift();
	double lg = log10(candidate->current.getLorentzFactor() * (1 + z));
	double lf = candidate->current.getLorentzFactor();

	int l = round((lg - lgmin) / (lgmax - lgmin) * (nlg - 1));  // index of closest tabulation point
	int key = Z*1e6 + (A-Z)*1e4 + (Z+dZ)*1e2 + (A+dA) - (Z+dZ);

	for (int i = 0; i < pdPhoton[key].size(); i++) {
		// check for random emission
		if (random.rand() > pdPhoton[key][i].emissionProbability[l])
			continue;

		// boost to lab frame
		double cosTheta = 2 * random.rand() - 1;
		double E = pdPhoton[key][i].energy * lf * (1 - cosTheta);
		candidate->addSecondary(22, E, pos);
	}
}

double PhotoDisintegration::lossLength(int id, double gamma, double z) {
	// check if nucleus
	if (not (isNucleus(id)))
		return std::numeric_limits<double>::max();

	int A = massNumber(id);
	int Z = chargeNumber(id);
	int N = A - Z;
	size_t idx = Z * 31 + N;

	// check if disintegration data available
	if ((Z > 26) or (N > 30))
		return std::numeric_limits<double>::max();
	const std::vector<double> &rate = pdRate[idx];
	if (rate.size() == 0)
		return std::numeric_limits<double>::max();

	// check if in tabulated energy range
	double lg = log10(gamma) * (1 + z);
	if ((lg <= lgmin) or (lg >= lgmax))
		return std::numeric_limits<double>::max();

	// total interaction rate
	double lossRate = interpolateEquidistant(lg, lgmin, lgmax, rate);

	// comological scaling, rate per physical distance
	lossRate *= pow(1 + z, 3) * photonFieldScaling(photonField, z);

	// average number of nucleons lost for all disintegration channels
	double avg_dA = 0;
	const std::vector<Branch> &branches = pdBranch[idx];
	for (size_t i = 0; i < branches.size(); i++) {
		int channel = branches[i].channel;
		int dA = 0;
		dA += 1 * digit(channel, 100000);
		dA += 1 * digit(channel, 10000);
		dA += 2 * digit(channel, 1000);
		dA += 3 * digit(channel, 100);
		dA += 3 * digit(channel, 10);
		dA += 4 * digit(channel, 1);

		double br = interpolateEquidistant(lg, lgmin, lgmax, branches[i].branchingRatio);
		avg_dA += br * dA;
	}

	lossRate *= avg_dA / A;
	return 1 / lossRate;
}

} // namespace crpropa