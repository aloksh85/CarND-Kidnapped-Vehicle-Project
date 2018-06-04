/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <assert.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles =50;
	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(x,std[0]);
	std::normal_distribution<double> dist_y(y,std[1]);
	std::normal_distribution<double> dist_theta(theta,std[2]);

	for (int i =0; i<num_particles; ++i) {
		weights.push_back(1.0);
		Particle p={i,dist_x(gen), dist_y(gen), dist_theta(gen), 1.0};
		particles.push_back(p);
	}

	is_initialized = true;

 cout<<"Initialization complete"<<endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	std::default_random_engine gen;
	for(auto& p:particles) {
		if(fabs(yaw_rate) > 0.0001){ // if yaw_rate is non-zero
			p.x+= ((velocity/yaw_rate)*(sin(p.theta+(yaw_rate*delta_t))-sin(p.theta)));
			p.y+= ((velocity/yaw_rate)*(cos(p.theta)-cos(p.theta+(yaw_rate*delta_t))));
			p.theta += (yaw_rate*delta_t);
		}
		else { // if yaw rate is zero
			p.x+=((velocity*cos(p.theta)*delta_t));
			p.y+=((velocity*sin(p.theta)*delta_t));

		}
		std::normal_distribution<double> dist_x(p.x,std_pos[0]);
		std::normal_distribution<double> dist_y(p.y,std_pos[1]);
		std::normal_distribution<double> dist_theta(p.theta,std_pos[2]);

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
																		 std::vector<LandmarkObs>& assoc_observations,
																		 const std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

		for(unsigned int i =0; i< predicted.size(); ++i) {
			LandmarkObs p = predicted.at(i);
			double min_dist = 100000;
			int min_idx = -1;
			for(unsigned int j =0; j< observations.size(); ++j) {
				LandmarkObs ob = observations.at(j);
				double distance = dist(p.x,p.y,ob.x,ob.y);
				if(distance < min_dist) {
				min_dist = distance;
				min_idx =j;
			}
		}
		if(min_idx >=0) {
		//DEBUG
		//	LandmarkObs a_ob = observations.at(min_idx);
		//	double a_dist = dist(p.x, p.y, a_ob.x, a_ob.y);
	  //		cout <<"pred to observation assoc: pred_x,y:"<<p.x<<","<<p.y<<", a_ob_x,y:"<<
	 //					a_ob.x<<","<<a_ob.y<<", dist: "<<a_dist<<endl;
			assoc_observations.push_back(observations.at(min_idx));
		}
	}
}

void computeLandmarkPredictions(std::vector<LandmarkObs>& predictions,
	const Particle& p, const double& sense_range, const Map& map_landmarks) {

	for (auto& landmark: map_landmarks.landmark_list) {
		double range = dist(landmark.x_f,landmark.y_f, p.x,p.y);

		if (range < sense_range) {
			LandmarkObs lobs;
			lobs.id = landmark.id_i;
			lobs.x = landmark.x_f;
			lobs.y = landmark.y_f;
			predictions.push_back(lobs);
		}
	}
}

void transformToMapCoordinates(std::vector<LandmarkObs>& transformed,
															 const std::vector<LandmarkObs>& original,
	 															double xp, double yp, double theta_p) {

	     for (const auto& og: original) {
				 LandmarkObs xformed;
				 xformed.x = (cos(theta_p)*og.x) - (sin(theta_p)*og.y) + xp;
				 xformed.y = (sin(theta_p)*og.x) + (cos(theta_p)*og.y) + yp;
				 xformed.id = og.id;
				 transformed .push_back(xformed);
			 }
		 }

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   3.33
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   http://planning.cs.uiuc.edu/node99.html

// for each particle
// predict all landmark measurements
// transform observations to map coordinates
// dataAssociation
// measurement probability
// weight update
	double weight_sum =0.0;
	for (unsigned int j =0; j<particles.size();++j) {
		Particle p = particles.at(j);
		std::vector<LandmarkObs> predictions;
		computeLandmarkPredictions(predictions, p, sensor_range, map_landmarks);
		std::vector<LandmarkObs> observations_map_coords;
		transformToMapCoordinates(observations_map_coords,observations, p.x,p.y,p.theta);
		std::vector<LandmarkObs> assoc_observations;
		dataAssociation(predictions, assoc_observations,observations_map_coords);
		assert(predictions.size() == assoc_observations.size());

		vector<int> assoc_ids;
		vector<double> assoc_x;
		vector<double> assoc_y;

		particles.at(j).weight =1.0;

		for(unsigned int i=0;i<predictions.size();++i) {
			LandmarkObs pred = predictions.at(i);
			LandmarkObs assoc_observation = assoc_observations.at(i);
			double x_diff_sq = pow(pred.x-assoc_observation.x,2);
			double y_diff_sq = pow(pred.y-assoc_observation.y,2);

			double meas_prob = (1/(2*M_PI*std_landmark[0]*std_landmark[1]))*std::exp(-x_diff_sq/(2*pow(std_landmark[0],2.0))+
																																											-y_diff_sq/(2*pow(std_landmark[1],2.0)));

      particles.at(j).weight*=meas_prob;
			assoc_ids.push_back(assoc_observation.id);
			assoc_x.push_back(assoc_observation.x);
			assoc_y.push_back(assoc_observation.y);
		}
		weights[j] =p.weight;
		weight_sum += p.weight;
		//cout<<"particle: "<<p.id<<", m prob: "<<p.weight<<endl;
		SetAssociations(p,assoc_ids, assoc_x, assoc_y);
	}

	double max_w =0.0;
	for( auto& w:weights) {
			w=w/weight_sum; // Normalize
			if (w > max_w)
				max_w = w;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	discrete_distribution<int> discrete_dist(weights.begin(),weights.end());
	vector<Particle> new_particles;

	for(int i =0; i<num_particles; ++i) {
		new_particles.push_back(particles[discrete_dist(gen)]);
	}
	particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
