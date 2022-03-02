// A version of gp_grid.stan that only estimates f = p0. Used for computing the Bayes factor
data {
  int<lower=1> n;                                         // No. of concentrations for drug 1
  vector[n] y;                                            // Viability measurements
  real x[n];                                              // Concentrations of drug 1
}
transformed data{
  vector[2] noise_hypers = [5,1]';                         // Hyperparameters for observation noise
  real lambda = 0.005;                                     // Real, standard deviation ratio between positive and negative controls
}
parameters {
  // For Non-Interaction
  real<lower=0,upper=1> la_1;
  real log10_ec50_1;
  real theta_1;
  real<lower=0> slope_1;

  // Variances
  real<lower=0> s;
  real<lower=0> s2_log10_ec50_1;

}
transformed parameters{
  vector<lower=0, upper=1>[n] p0;          // Monotherapy drug 2
  for (j in 1:n){
    p0[j] = la_1+(1-la_1)/(1+10^(slope_1*(x[j]-log10_ec50_1)));
  }
}
model {
  vector[n] noise;
  // Variances
  target += inv_gamma_lpdf(s | noise_hypers[1],noise_hypers[2]);
  target += inv_gamma_lpdf(s2_log10_ec50_1 | 3,2);


  // Monotherapies
  target += beta_lpdf(la_1 | 1,1.25);
  target += gamma_lpdf(slope_1 | 1,1);
  target += std_normal_lpdf(theta_1);
  target += normal_lpdf(log10_ec50_1 | theta_1,sqrt(s2_log10_ec50_1));

  // Response
  noise = s*sqrt((p0+lambda));
  target += normal_lpdf(y | p0,noise);

}
generated quantities {
  real ec50_1;
  // Getting EC50 on original scale
  ec50_1 = 10^log10_ec50_1;
}
