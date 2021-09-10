/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;

static default_random_engine gen;

#define EPS 0.00001

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
    // TODO: Set the number of particles
  
  if (is_initialized)
  {
    // Step over the init phase
    return;
  }

  // Init variables
  num_particles = 100;
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  // Get distributions
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  // Generate particles with normal distribution with mean on GPS values.
  for (int a = 0; a < num_particles; a++)
  {
    Particle part;
    part.id = a;
    part.x = dist_x(gen);
    part.y = dist_y(gen);
    part.theta = dist_theta(gen);
    part.weight = 1.0;
  
    particles.push_back(part);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  // Creating normal distributions
  normal_distribution<double> dist_x(0, std_x);
  normal_distribution<double> dist_y(0, std_y);
  normal_distribution<double> dist_theta(0, std_theta);

  // Calculate new state.
  for (int a = 0; a < num_particles; a++)
  {
    double theta = particles[a].theta;

    if (fabs(yaw_rate) < EPS)
    { 
      // When yaw is not changing.
      particles[a].x += velocity * delta_t * cos(theta);
      particles[a].y += velocity * delta_t * sin(theta);
    }
    else
    {
      particles[a].x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
      particles[a].y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
      particles[a].theta += yaw_rate * delta_t;
    }

    // Adding the generated distributions to the particles
    particles[a].x += dist_x(gen);
    particles[a].y += dist_y(gen);
    particles[a].theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
   for (unsigned int a = 0; a < observations.size(); a++)
  { 
    // Init vars (very big and invalid value)
    double minDist = 10000000;
    int newMapID = 0;

    // For each predition.
    for (unsigned b = 0; b < predicted.size(); b++)
    {
      double Dist_x = observations[a].x - predicted[b].x;
      double Dist_y = observations[a].y - predicted[b].y;
      double Dist = Dist_x * Dist_x + Dist_y * Dist_y;

      // Update min
      if (Dist < minDist)
      {
        minDist = Dist;
        newMapID = predicted[b].id;
      }
    }
    // Update id with the found newMapID
    observations[a].id = newMapID;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double stdLandmarkRange = std_landmark[0];
  double stdLandmarkBearing = std_landmark[1];

  for (int a = 0; a < num_particles; a++)
  {
    double x = particles[a].x;
    double y = particles[a].y;
    double theta = particles[a].theta;

    // Find landmarks in given range
    double sensor_range_x2 = sensor_range * sensor_range;
    vector<LandmarkObs> inRangeLandmarks;
    for (unsigned int b = 0; b < map_landmarks.landmark_list.size(); b++)
    {
      float landmarkX = map_landmarks.landmark_list[b].x_f;
      float landmarkY = map_landmarks.landmark_list[b].y_f;
      int id = map_landmarks.landmark_list[b].id_i;
      double dX = x - landmarkX;
      double dY = y - landmarkY;
      if (dX * dX + dY * dY <= sensor_range_x2)
      {
        inRangeLandmarks.push_back(LandmarkObs{id, landmarkX, landmarkY});
      }
    }

    // Transform observation coordinates.
    vector<LandmarkObs> mappedObservations;
    for (unsigned int b = 0; b < observations.size(); b++)
    {
      double xx = cos(theta) * observations[b].x - sin(theta) * observations[b].y + x;
      double yy = sin(theta) * observations[b].x + cos(theta) * observations[b].y + y;
      mappedObservations.push_back(LandmarkObs{observations[b].id, xx, yy});
    }

    // Observation association to landmark.
    dataAssociation(inRangeLandmarks, mappedObservations);

    // Reseting weight.
    particles[a].weight = 1.0;
    // Calculate weights.
    for (unsigned int b = 0; b < mappedObservations.size(); b++)
    {
      double observationX = mappedObservations[b].x;
      double observationY = mappedObservations[b].y;

      int landmarkId = mappedObservations[b].id;

      double landmarkX, landmarkY;
      unsigned int k = 0;
      unsigned int nLandmarks = inRangeLandmarks.size();
      bool found = false;
      while (!found && k < nLandmarks)
      {
        if (inRangeLandmarks[k].id == landmarkId)
        {
          found = true;
          landmarkX = inRangeLandmarks[k].x;
          landmarkY = inRangeLandmarks[k].y;
        }
        k++;
      }

      // Calc weight
      double dX = observationX - landmarkX;
      double dY = observationY - landmarkY;

      double weight = (1 / (2 * M_PI * stdLandmarkRange * stdLandmarkBearing)) * exp(-(dX * dX / (2 * stdLandmarkRange * stdLandmarkRange) + (dY * dY / (2 * stdLandmarkBearing * stdLandmarkBearing))));
      if (weight == 0)
      {
        particles[a].weight *= EPS;
      }
      else
      {
        particles[a].weight *= weight;
      }
    }
  }

  

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<double> weights;
  double maxWeight = numeric_limits<double>::min();
  for (int a = 0; a < num_particles; a++)
  {
    weights.push_back(particles[a].weight);
    if (particles[a].weight > maxWeight)
    {
      maxWeight = particles[a].weight;
    }
  }

  // Generating distributions
  uniform_real_distribution<double> distribution_real(0.0, maxWeight);
  uniform_int_distribution<int> distribution_int(0, num_particles - 1);

  int index = distribution_int(gen);
  double beta = 0.0;

  // Apply distributions
  vector<Particle> resampledParticles;
  for (int a = 0; a < num_particles; a++)
  {
    beta += distribution_real(gen) * 2.0;
    while (beta > weights[index])
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampledParticles.push_back(particles[index]);
  }

  particles = resampledParticles;
  
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  
    //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();
  
  
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}